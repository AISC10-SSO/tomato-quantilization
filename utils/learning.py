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

class TomatoGridWrapper():
    def __init__(self, config: dict):
        """
        Wrapper for the TomatoGrid class, to make it compatible with gymnasium
        Currently not compatible with gymnasium, as gymnasium seems to be incompatible with python 3.13
        """

        super().__init__()
        self.gridworld = TomatoGrid(**config)

        """
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(6, 7, 9),
            dtype=np.uint8
        )
        """

    def reset(self, seed: int|None = None, options: dict|None = None):
        if seed is not None:
            self.gridworld.seed = seed
            self.gridworld.reset(reset_rng=True)
        else:
            self.gridworld.reset()

        observation = self.gridworld.get_state_tensor(format="numpy")
        valid_actions = self.gridworld.get_valid_actions()

        return observation, {"valid_actions": valid_actions}

    def step(self, action: int):

        self.gridworld.update_grid(list(Action)[action])

        observation = self.gridworld.get_state_tensor(format="numpy")
        valid_actions = self.gridworld.get_valid_actions()
        output = self.gridworld.get_current_utility()
        reward = output["misspecified_reward"]
        terminated = self.gridworld.is_terminal
        truncated = False
        info = {"true_utility": output["true_utility"], "valid_actions": valid_actions}

        # observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
    
    def render(self):
        state: list[list[str]] = self.gridworld.grid_state
        state[self.gridworld.agent_position[0]][self.gridworld.agent_position[1]] = "A"

        for row in state:
            print("".join(row))

class ResidualConvLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)

        self.linear = nn.Linear(input_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.mean(dim=(-2,-1))

        return F.relu(self.bn(self.conv(x))) + self.linear(residual).unsqueeze(-1).unsqueeze(-1)
    
class ResidualFeedForwardLayer(nn.Module):
    def __init__(self, residual_dim: int, feedforward_dim: int):
        super().__init__()

        self.W_up = nn.Linear(residual_dim, feedforward_dim)
        self.W_down = nn.Linear(feedforward_dim, residual_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.relu(self.W_up(x))) + x

class QNetwork(nn.Module):
    def __init__(
            self, *,
            input_channels: int,
            action_size: int,
            model_kl: bool = False,
            network_widths: tuple[int, int] = (16, 32)):
        """
        Convolutional Q Network for the tomato gridworld
        """
        super().__init__()

        self.conv1 = ResidualConvLayer(input_channels, network_widths[0])
        self.conv2 = ResidualConvLayer(network_widths[0], network_widths[1])
        self.feedforward = ResidualFeedForwardLayer(network_widths[1], network_widths[1] * 4)

        self.fc_q = nn.Linear(network_widths[1], action_size)
        self.model_kl = model_kl

        if self.model_kl:
            self.fc_kl = nn.Linear(network_widths[1], action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q Network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size)
        """

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.mean(dim=(-2,-1))

        x = self.feedforward(x)

        output = {"q_reward": self.fc_q(x)}

        if self.model_kl:
            output["q_kl"] = self.fc_kl(x)

        return output

    
class QAgent(nn.Module):
    def __init__(
            self, *,
            input_channels: int = 6,
            action_size: int = 5,
            gamma: float = 0.99,
            kl_divergence_coefficient: float|None|Literal["auto"] = None,
            t_inv_sample: float|None|Literal["auto"] = None,
            t_inv_deploy: float|None = None,
            reward_cap: float|None = None,
            q_cap: float|None = None,
            variable_t_inv: bool = False,
            double_network: bool = True,
            network_widths: tuple[int, int] = (32, 64)):
    
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

        network_kwargs = {"input_channels": input_channels, "action_size": action_size, "model_kl": self.model_kl, "network_widths": network_widths}

        self.networks = nn.ModuleDict({
            str(i): QNetwork(**network_kwargs)
            for i in self.network_target_map.keys()
        })
        self.target_networks = nn.ModuleDict({
            str(i): QNetwork(**network_kwargs)
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

    def get_loss(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

        loss = torch.tensor(0.0)

        actions = data["action"].unsqueeze(1)

        for network_idx, target_network_idx in self.network_target_map.items():
            for name, output in outputs[network_idx].items():
                try:
                    output_gathered = output.gather(1, index=actions).squeeze(1)
                    loss += F.smooth_l1_loss(
                        output_gathered,
                        targets[name][target_network_idx])
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
    
    def update_target_networks(self, tau: float = 0.5) -> None:
        """
        Update the target network with exponential moving average, using the current network's parameters

        Args:
            tau (float): Tau value for the exponential moving average
        """

        for network_idx, target_network_idx in self.network_target_map.items():
            new_state_dict = self.networks[network_idx].state_dict()
            for name, param in new_state_dict.items():
                self.target_networks[target_network_idx].state_dict()[name].copy_(
                    self.target_networks[target_network_idx].state_dict()[name] * (1-tau) + param * tau)
                
        return
        
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
        # self.average_reward = new_reward if self.average_reward == 0 else self.average_reward * (1-tau) + new_reward * tau
        pass
    
    def update_average_kl_divergence(self, new_kl_divergence: float, tau: float = 0.5):
        # self.average_kl_divergence = new_kl_divergence if self.average_kl_divergence == 0 else self.average_kl_divergence * (1-tau) + new_kl_divergence * tau
        pass

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

        self.gridworld = TomatoGridWrapper(config=self.gridworld_config)

        self.q_agent = QAgent(**q_agent_config)

        self.optimizer = torch.optim.AdamW(self.q_agent.parameters(), **adamw_config)

        self.outputs = []

    def _get_action_validity(self, valid_actions: list[Action]):
        action_validity = torch.tensor([action in valid_actions for action in list(Action)])
        return action_validity
    
    def train(
            self,
            steps: int,
            update_interval: int = 100,
            test_interval: int = 1000):
        
        self.gradient_descent_steps = 0
        
        observation, info = self.gridworld.reset()
        action_validity = self._get_action_validity(info["valid_actions"])

        for step_idx in tqdm(range(steps)):
            dict_ = {
                "state": torch.tensor(observation).type(torch.float32),
                "action_validity": action_validity
            }

            action_idx = self.q_agent.get_action(
                state = dict_["state"].unsqueeze(0),
                action_validity=dict_["action_validity"].unsqueeze(0),
                mode="sample")

            observation, reward, terminal, _, info = self.gridworld.step(action_idx)
            action_validity = self._get_action_validity(info["valid_actions"])

            dict_["next_state"] = torch.tensor(observation).type(torch.float32)
            dict_["next_state_action_validity"] = action_validity
            dict_["action"] = torch.tensor(action_idx)
            dict_["reward"] = torch.tensor(reward)

            self.state_buffer.add(dict_)

            if terminal:
                observation, info = self.gridworld.reset()
                action_validity = self._get_action_validity(info["valid_actions"])

            if step_idx % update_interval == 0 and step_idx > 0 and len(self.state_buffer) > self.config["batch_size"]:
                for _ in range(update_interval):
                    loss_output = self.q_agent.get_loss(self.state_buffer.get_batch())

                    self.optimizer.zero_grad()
                    loss = loss_output["loss"]

                    """
                    if self.config["kl_divergence_target"] is not None:
                        kl_divergence = loss_output["kl_divergence"]
                        kl_divergence_loss = F.smooth_l1_loss(
                            kl_divergence.mean() / self.config["kl_divergence_target"], torch.tensor(1))
                        loss += kl_divergence_loss
                    """

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.q_agent.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    self.gradient_descent_steps += 1
                
                    self.q_agent.update_target_networks(tau=0.01)

            if step_idx % test_interval == 0:
                test_output = self.test_model()
                self.outputs.append(test_output)

    def test_model(self) -> dict[str, float]:
        gridworlds = [TomatoGridWrapper(self.gridworld_config) for _ in range(25)]

        output_tuples = [gridworld.reset() for gridworld in gridworlds]

        misspecified_rewards = []
        true_utilities = []

        state = torch.stack([torch.tensor(output_tuple[0]).type(torch.float32) for output_tuple in output_tuples])
        action_validity = torch.stack([self._get_action_validity(output_tuple[1]["valid_actions"]) for output_tuple in output_tuples])

        for _ in range(100):
            action_indices = self.q_agent.get_action(state = state, action_validity=action_validity, mode="deploy")

            output_tuples = [gridworld.step(action_idx) for gridworld, action_idx in zip(gridworlds, action_indices)]

            state = torch.stack([torch.tensor(output_tuple[0]).type(torch.float32) for output_tuple in output_tuples])
            action_validity = torch.stack([self._get_action_validity(output_tuple[-1]["valid_actions"]) for output_tuple in output_tuples])

            misspecified_rewards += [output_tuple[1] for output_tuple in output_tuples]
            true_utilities += [output_tuple[-1]["true_utility"] for output_tuple in output_tuples]

        misspecified_reward = np.mean(misspecified_rewards)
        true_utility = np.mean(true_utilities)

        return {"step": self.gradient_descent_steps, "misspecified_reward": misspecified_reward, "true_utility": true_utility}