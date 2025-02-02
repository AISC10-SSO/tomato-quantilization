import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import random
import logging

from typing import Literal
from collections import deque
from utils import TomatoGrid, Action
import utils.functions as UF

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):

    def __init__(self, input_channels: int, action_size: int, model_kl: bool = False):
        """
        Convolutional Q Network for the tomato gridworld
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 512)
        self.fc_q = nn.Linear(512, action_size)

        self.model_kl = model_kl

        if self.model_kl:
            self.fc_kl = nn.Linear(512, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q Network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=(-2,-1))

        x = F.relu(self.fc1(x))

        output = {"q_reward": self.fc_q(x)}

        if self.model_kl:
            output["q_kl"] = self.fc_kl(x)

        return output

    
class QAgent(nn.Module):
    def __init__(
            self, *,
            input_channels: int = 6,
            action_size: int = 5,
            gamma: float = 1.99,
            kl_divergence_coefficient: float|None|Literal["auto"] = None,
            t_inv_sample: float|None|Literal["auto"] = None,
            t_inv_deploy: float|None = None,
            reward_cap: float|None = None,
            q_cap: float|None = None,
            variable_t_inv: bool = False,
            double_network: bool = True):
    
        super().__init__()

        self.input_channels = input_channels
        self.action_size = action_size
        self.model_kl = kl_divergence_coefficient is not None

        self.double_network = double_network

        if self.double_network:
            self.network_target_map = {
                "1": "2",
                "2": "1"
            }
        else:
            self.network_target_map = {
                "1": "1"
            }

        self.networks = nn.ModuleDict({
            str(i): QNetwork(input_channels, action_size, model_kl=self.model_kl)
            for i in self.network_target_map.keys()
        })
        self.target_networks = nn.ModuleDict({
            str(i): QNetwork(input_channels, action_size, model_kl=self.model_kl)
            for i in self.networks.keys()
        })

        for i in self.networks.keys():
            self.target_networks[str(i)].load_state_dict(self.networks[str(i)].state_dict())

        self.gamma = gamma

        self.t_inv_sample = t_inv_sample

        if variable_t_inv:
            self.t_inv_deploy = nn.Parameter(torch.tensor(t_inv_deploy))
        else:
            self.t_inv_deploy = t_inv_deploy

        self.reward_cap = reward_cap

        self.q_cap = None if q_cap is None else (1/(1-gamma)) * q_cap

        self.kl_divergence_coefficient = kl_divergence_coefficient

        self.average_reward = 0
        self.average_kl_divergence = 0

        self.print_info = False

    def get_loss(self, data: dict[str, torch.Tensor]):
        """
        Get the loss from a batch of data, containing rewards per state, actions taken, and input states

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing "state", "next_state", "reward", "action", and "action_validity"
            t_inv (float): inverse T value for the temperature scaling

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "loss", "outputs", "probabilities", and "kl_divergence"
        """

        # Predict the rewards for each state, given the t_inv value
        # This means taking the probabilities of each action and multiplying them by the predicted rewards (Q values)
        # Throw away the first state as we don't know what came before it

        invalid_actions_mask = (~data["action_validity"]).float() * -1e9
        next_state_invalid_actions_mask = (~data["next_state_action_validity"]).float() * -1e9

        with torch.no_grad():
            # Get Q-values from both target network
            next_outputs = {
                k: target_network(data["next_state"])
                for k, target_network in self.target_networks.items()
            }
            next_probabilities = {
                k: self.get_probabilities(next_output, mode="deploy", action_validity=data["next_state_action_validity"])
                for k, next_output in next_outputs.items()
            }
            next_predictions = {
                network_k: {
                    output_k: torch.einsum("ba,ba->b", next_probability, next_output)
                    for output_k, next_output in next_outputs[network_k].items()
                }
                for network_k, next_probability in next_probabilities.items()
            }

            reward = data["reward"].float()

            if self.reward_cap is not None:
                reward = reward.clamp(max=self.reward_cap)

            reward_centered = reward - self.average_reward

            targets = {
                "q_reward": {
                    k: reward_centered + self.gamma * next_prediction["q_reward"]
                    for k, next_prediction in next_predictions.items()
                }
            }

            self.update_average_reward(reward.mean().item())

            if self.model_kl:

                base_probabilities = F.softmax(next_state_invalid_actions_mask, dim=-1)

                kl_divergences = {
                    k: UF.safe_kl_div(base_probabilities=base_probabilities, altered_probabilities=action_probabilities)
                    for k, action_probabilities in next_probabilities.items()
                }

                kl_divergences_centered = {
                    k: kl_divergence - self.average_kl_divergence
                    for k, kl_divergence in kl_divergences.items()
                }

                targets["q_kl"] = {
                    k: kl_divergences_centered[k] + self.gamma * next_predictions[k]["q_kl"]
                    for k in self.networks.keys()
                }

                mean_kl_divergence = torch.cat(list(kl_divergences.values())).mean().item()
                self.update_average_kl_divergence(mean_kl_divergence)

        outputs = {
            k: self.networks[k](data["state"])
            for k in self.networks.keys()
        }

        loss = torch.tensor(0)

        for network_idx, target_network_idx in self.network_target_map.items():
            for name, output in outputs[network_idx].items():
                try:
                    loss += F.smooth_l1_loss(output.gather(1, index=data["action"].long().unsqueeze(1)), targets[name][target_network_idx])
                except Exception as e:
                    print(f"{output.shape=}, {data['action'].shape=}")
                    raise e

        # Use average of both networks for probabilities and outputs
        probabilities = {
            k: self.get_probabilities(outputs[k], mode="deploy", action_validity=data["action_validity"])
            for k in outputs.keys()
        }

        mean_probabilities = torch.mean(torch.stack(list(probabilities.values()), dim=0), dim=0)

        kl_divergence = UF.safe_kl_div(
            base_probabilities=F.softmax(invalid_actions_mask, dim=-1).detach(),
            altered_probabilities=mean_probabilities)

        return {
            "loss": loss,
            "outputs": outputs,
            "probabilities": mean_probabilities,
            "kl_divergence": kl_divergence
        }

    def get_action(
            self, *,
            state: torch.Tensor,
            action_validity: torch.Tensor|None = None,
            mode: Literal["sample", "deploy"] = "sample") -> int:
        """
        Get the action to take from the network

        Args:
            state (torch.Tensor): Input state of shape (input_channels, height, width)
            action_validity (torch.Tensor|None): Action validity mask of shape (action_size)

        Returns:
            int: Action to take
        """
        probabilities = {
            k: self.get_probabilities(network(state), mode=mode, action_validity=action_validity)
            for k, network in self.networks.items()
        }

        mean_probabilities = torch.mean(torch.stack(list(probabilities.values()), dim=0), dim=0)
        # Choose randomly

        try:
            output =  torch.multinomial(mean_probabilities, 1)
        except Exception as e:
            logger.warning(f"Error getting action: {e}")
            output = torch.argmax(mean_probabilities, dim=-1)

        if output.shape[0] == 1:
            return output.item()
        else:
            return output.flatten().tolist()
    
    def update_target_network(self, tau: float = 0.5):
        """
        Update the target network with exponential moving average, using the current network's parameters

        Args:
            tau (float): Tau value for the exponential moving average
        """

        new_state_dict = self.network_1.state_dict()
        for name, param in new_state_dict.items():
            self.target_network_1.state_dict()[name].copy_(
                self.target_network_1.state_dict()[name] * (1-tau) + param * tau)
        
        new_state_dict = self.network_2.state_dict()
        for name, param in new_state_dict.items():
            self.target_network_2.state_dict()[name].copy_(
                self.target_network_2.state_dict()[name] * (1-tau) + param * tau)
        
    def get_probabilities(self, outputs: dict[str, torch.Tensor], mode: Literal["sample", "deploy"], action_validity: torch.Tensor|None = None):
        """
        Get the probabilities from the network, can use softmax, max, or capped softmax

        Args:
            outputs (dict[str, torch.Tensor]): Outputs from the network of shape (batch_size, action_size)
            mode (Literal["sample", "train", "deploy"]): Mode to use for the softmax
            action_validity (torch.Tensor|None): Action validity mask of shape (batch_size, action_size)

        Other Dependencies:
            self.t_inv_sample for t_inv values (if None, it uses argmax, and self.q_cap is ignored)
            self.t_inv_deploy for t_inv values (if None, it uses argmax, and self.q_cap is ignored)
        Returns:
            torch.Tensor: Probabilities of shape (batch_size, action_size)
        """

        # Mask out invalid actions
        if mode == "sample" and self.t_inv_sample != "auto":
            t_inv = self.t_inv_sample
        else:
            t_inv = self.t_inv_deploy

        if action_validity is None:
            action_validity_mask = torch.zeros_like(outputs["q_reward"])
        else:
            action_validity_mask = (~action_validity).float() * -1e9

        average_q = (self.average_reward or 0) / (1-self.gamma)
        adjusted_q_outputs = outputs["q_reward"] + average_q

        if self.print_info:
            print(f"Average q: {average_q}")
            print(f"Average non-adjusted output: {outputs['q'].mean()}")
            print(f"Average output: {adjusted_q_outputs.mean()}")

        # Apply temperature scaling if t_inv is provided
        if t_inv is None:
            # If no t_inv, use argmax
            logits = torch.full_like(outputs, -1e9)
            logits[torch.arange(outputs.shape[0]), torch.argmax(adjusted_q_outputs + action_validity_mask, dim=-1)] = 0
        else:
            if self.q_cap is None:
                # If no q_cap, use softmax
                logits = adjusted_q_outputs * t_inv + action_validity_mask
            else:
                # If q_cap is provided, use capped softmax
                logits = -F.softplus(t_inv * (self.q_cap - adjusted_q_outputs)) + action_validity_mask

        if self.model_kl:
            # Use the expected future KL divergence to scale the probabilities
            # Less KL divergence = more likely to take the action
            logits -= outputs["q_kl"] * self.kl_divergence_coefficient

        return F.softmax(logits, dim=-1)
    

    def update_average_reward(self, new_reward: float, tau: float = 0.5):
        self.average_reward = new_reward if self.average_reward == 0 else self.average_reward * (1-tau) + new_reward * tau
    
    def update_average_kl_divergence(self, new_kl_divergence: float, tau: float = 0.5):
        self.average_kl_divergence = new_kl_divergence if self.average_kl_divergence == 0 else self.average_kl_divergence * (1-tau) + new_kl_divergence * tau

    def get_kl_divergence_coefficient(self):
        if self.kl_divergence_coefficient is None:
            return None
        
        if self.kl_divergence_coefficient != "auto":
            return min(self.kl_divergence_coefficient, 1e2)
        
        t_inv = self.t_inv_deploy

        if type(t_inv) == torch.Tensor:
            t_inv = t_inv

        return 1/max(t_inv, 1e-2)

   
class StateBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffers: deque[dict[str, torch.Tensor]] = deque(maxlen=buffer_size)

    def add(self, data: dict[str, torch.Tensor]):
        self.buffers.append(data)

    def get_batch(self) -> dict[str, torch.Tensor]:

        if len(self.buffers) < self.batch_size:
            raise ValueError("Not enough data in buffer to get a batch")
        
        batch = random.sample(self.buffers, self.batch_size)
        # Convert list of dicts to dict of lists
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = torch.stack([d[key].squeeze() for d in batch])

        return batch_dict
    
    def __len__(self):
        return len(self.buffers)
    
    def clear(self):
        self.buffers.clear()

class QLearning:
    def __init__(
            self, *,
            config: dict,
            adamw_config: dict,
            gridworld_config: dict = {},
            q_agent_config: dict = {}):

        self.config = config
        self.state_buffer = StateBuffer(self.config["buffer_size"], self.config["batch_size"])

        self.gridworld_config = gridworld_config

        self.q_agent = QAgent(**q_agent_config)

        self.optimizer = torch.optim.AdamW(self.q_agent.parameters(), **adamw_config)

        self.outputs = []

    def get_action_validity(self, gridworld: TomatoGrid):
        valid_actions = gridworld.get_valid_actions()
        action_validity = torch.tensor([action in valid_actions for action in list(Action)])
        return action_validity
    
    def train(
            self,
            steps: int):
        
        gridworld = TomatoGrid(**self.gridworld_config)

        for step_idx in tqdm(range(steps)):
            dict_ = {}
            
            state = gridworld.get_state_tensor()
            dict_["state"] = state

            dict_["action_validity"] = self.get_action_validity(gridworld).unsqueeze(0)

            action_idx = self.q_agent.get_action(state = state.unsqueeze(0), action_validity=dict_["action_validity"], mode="sample")
            dict_["action"] = torch.tensor(action_idx)

            action = list(Action)[action_idx]
            output = gridworld.update_grid(action)

            dict_["next_state"] = gridworld.get_state_tensor()
            dict_["next_state_action_validity"] = self.get_action_validity(gridworld).unsqueeze(0)

            dict_["reward"] = torch.tensor(output.misspecified_reward)
            self.state_buffer.add(dict_)

            if step_idx % 100 == 0 and step_idx > 0 and len(self.state_buffer) > self.config["batch_size"]:
                for _ in range(10):
                    loss_output = self.q_agent.get_loss(self.state_buffer.get_batch())

                    self.optimizer.zero_grad()
                    loss = loss_output["loss"]

                    if self.config["kl_divergence_target"] is not None:
                        kl_divergence = loss_output["kl_divergence"]
                        kl_divergence_loss = F.smooth_l1_loss(
                            kl_divergence.mean() / self.config["kl_divergence_target"], torch.tensor(1))
                        loss += kl_divergence_loss

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.q_agent.parameters(), max_norm=1.0)

                    self.optimizer.step()
                
                    self.q_agent.update_target_network(tau=0.01)

            if step_idx % 1000 == 0:
                test_output = self.test_model()
                self.outputs.append(test_output)
                """
                print(test_output)
                print(f"Average reward: {self.q_agent.average_reward}")
                print(f"Average kl divergence: {self.q_agent.average_kl_divergence}")
                """

    def test_model(self):
        gridworlds = [TomatoGrid(**self.gridworld_config) for _ in range(10)]

        outputs = []

        for _ in range(100):
            state = torch.stack([gridworld.get_state_tensor() for gridworld in gridworlds])
            action_validity = torch.stack([self.get_action_validity(gridworld) for gridworld in gridworlds])
            action_indices = self.q_agent.get_action(state = state, action_validity=action_validity, mode="deploy")
            actions = [list(Action)[action_idx] for action_idx in action_indices]
            outputs += [gridworld.update_grid(action) for gridworld, action in zip(gridworlds, actions)]

        misspecified_reward = np.mean([output.misspecified_reward for output in outputs])
        true_utility = np.mean([output.true_utility for output in outputs])

        return {"misspecified_reward": misspecified_reward, "true_utility": true_utility}
