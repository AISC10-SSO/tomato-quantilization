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

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):

    def __init__(self, input_channels: int, action_size: int):
        """
        Convolutional Q Network for the tomato gridworld
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, action_size)

        self.reward_so_far_projection = nn.Linear(1, 64, bias=False)

    def forward(self, x: torch.Tensor, reward_so_far: torch.Tensor|None = None) -> torch.Tensor:
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

        if reward_so_far is not None:
            x = x + self.reward_so_far_projection(reward_so_far)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
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
            variable_t_inv: bool = False):
    
        super().__init__()

        self.input_channels = input_channels
        self.action_size = action_size

        self.network_1 = QNetwork(input_channels, action_size)
        self.network_2 = QNetwork(input_channels, action_size)

        self.gamma = gamma

        self.t_inv_sample = t_inv_sample

        if variable_t_inv:
            self.t_inv_deploy = nn.Parameter(torch.tensor(t_inv_deploy))
        else:
            self.t_inv_deploy = t_inv_deploy

        self.reward_cap = reward_cap

        self.q_cap = None if q_cap is None else (1/(1-gamma)) * q_cap

        # Create two target networks
        self.target_network_1 = QNetwork(input_channels, action_size)
        self.target_network_2 = QNetwork(input_channels, action_size)
        
        self.target_network_1.load_state_dict(self.network_1.state_dict())
        self.target_network_2.load_state_dict(self.network_2.state_dict())

        self.kl_divergence_coefficient = kl_divergence_coefficient

        self.average_reward = 0
        self.average_kl_divergence = 0

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
            # Get Q-values from both target networks
            next_q_values_1 = self.target_network_1(data["next_state"])
            next_q_values_2 = self.target_network_2(data["next_state"])
            
            next_probabilities_1 = self.get_probabilities(
                next_q_values_1 + next_state_invalid_actions_mask,
                mode="deploy",
                action_validity=data["action_validity"])
            
            next_probabilities_2 = self.get_probabilities(
                next_q_values_2 + next_state_invalid_actions_mask,
                mode="deploy",
                action_validity=data["action_validity"])

            next_values_1 = torch.einsum("ba,ba->b", next_probabilities_1, next_q_values_1 + self.average_reward)
            next_values_2 = torch.einsum("ba,ba->b", next_probabilities_2, next_q_values_2 + self.average_reward)

            reward = data["reward"]

            if self.reward_cap is not None:
                reward = reward.clamp(max=self.reward_cap)

            target_rewards_1 = reward + self.gamma * next_values_1
            target_rewards_2 = reward + self.gamma * next_values_2

            average_reward = torch.mean(reward.float()).item()

            if (kl_coefficient := self.get_kl_divergence_coefficient()) is not None:
                base_probabilities = F.softmax(next_state_invalid_actions_mask, dim=-1)

                next_kl_divergence_1 = self.safe_kl_div(base_probabilities=base_probabilities, altered_probabilities=next_probabilities_1)
                next_kl_divergence_2 = self.safe_kl_div(base_probabilities=base_probabilities, altered_probabilities=next_probabilities_2)

                target_rewards_1 = target_rewards_1 - next_kl_divergence_1 * kl_coefficient
                target_rewards_2 = target_rewards_2 - next_kl_divergence_2 * kl_coefficient

        self.update_average_reward(average_reward)

        # Calculate losses for both networks
        outputs_1 = self.network_1(data["state"])
        outputs_2 = self.network_2(data["state"])

        predicted_rewards_1 = outputs_1.gather(1, data["action"].unsqueeze(1)).squeeze() + self.average_reward
        predicted_rewards_2 = outputs_2.gather(1, data["action"].unsqueeze(1)).squeeze() + self.average_reward

        # Swap the target rewards for the two networks
        loss_1 = F.smooth_l1_loss(predicted_rewards_1, target_rewards_2)
        loss_2 = F.smooth_l1_loss(predicted_rewards_2, target_rewards_1)

        # Total loss is the sum of both networks' losses
        loss = loss_1 + loss_2

        # Use average of both networks for probabilities and outputs
        outputs = (outputs_1 + outputs_2) / 2

        probabilities = self.get_probabilities(outputs.detach(), mode="deploy", action_validity=data["action_validity"])

        kl_divergence = self.safe_kl_div(base_probabilities=F.softmax(invalid_actions_mask, dim=-1).detach(), altered_probabilities=probabilities)

        self.update_average_kl_divergence(kl_divergence.mean().item())

        return {
            "loss": loss,
            "outputs": outputs,
            "probabilities": probabilities,
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
        outputs = (self.network_1(state) + self.network_2(state)) / 2
        probabilities = self.get_probabilities(outputs, mode=mode, action_validity=action_validity)
        # Choose randomly

        try:
            output =  torch.multinomial(probabilities, 1)
        except Exception as e:
            logger.warning(f"Error getting action: {e}")
            output = torch.argmax(probabilities, dim=-1)

        if output.shape[0] == 1:
            return output.item()
        else:
            return output.flatten().tolist()
    
    def update_target_network(self, tau: float):
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
        
    def get_probabilities(self, outputs: torch.Tensor, mode: Literal["sample", "deploy"], action_validity: torch.Tensor|None = None):
        """
        Get the probabilities from the network, can use softmax, max, or capped softmax

        Args:
            outputs (torch.Tensor): Outputs from the network of shape (batch_size, action_size)
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
            action_validity_mask = torch.zeros_like(outputs)
        else:
            action_validity_mask = (~action_validity).float() * -1e9

        average_q = self.average_reward or 0 / (1-self.gamma)
        adjusted_outputs = outputs + average_q

        # Q cap should be adjusted by the penalized kl divergence
        average_kl_divergence = self.average_kl_divergence or 0 / (1-self.gamma)
        if (kl_coefficient := self.get_kl_divergence_coefficient()) is not None: # I am the walrus
            adjusted_outputs = adjusted_outputs + average_kl_divergence * kl_coefficient


        # Apply temperature scaling if t_inv is provided
        if t_inv is None:
            # If no t_inv, use argmax
            probabilities = torch.zeros_like(outputs)
            probabilities[torch.arange(outputs.shape[0]), torch.argmax(adjusted_outputs + action_validity_mask, dim=-1)] = 1
        else:
            if self.q_cap is None:
                # If no q_cap, use softmax
                probabilities = F.softmax(adjusted_outputs * t_inv + action_validity_mask, dim=-1)
            else:
                # If q_cap is provided, use capped softmax
                log_probabilities = -self.safe_log_one_plus_exp(t_inv * (self.q_cap - adjusted_outputs)) + action_validity_mask
                probabilities = F.softmax(log_probabilities, dim=-1)

        # Clamp negative probabilities
        if probabilities.max() < 0:
            logger.warning("Negative probability detected")
            probabilities = probabilities.clamp(min=0)

        # Clamp NaN or inf probabilities
        if probabilities.isnan().any() or probabilities.isinf().any():
            logger.warning("NaN or inf probability detected")
            probabilities = torch.ones_like(probabilities)

        return probabilities
    

    def update_average_reward(self, new_reward: float):
        self.average_reward = new_reward if self.average_reward == 0 else self.average_reward * 0.9 + new_reward * 0.1
    
    def update_average_kl_divergence(self, new_kl_divergence: float):
        self.average_kl_divergence = new_kl_divergence if self.average_kl_divergence == 0 else self.average_kl_divergence * 0.9 + new_kl_divergence * 0.1

    def get_kl_divergence_coefficient(self):
        if self.kl_divergence_coefficient is None:
            return None
        
        if self.kl_divergence_coefficient != "auto":
            return max(self.kl_divergence_coefficient, 1e2)
        
        t_inv = self.t_inv_deploy

        if type(t_inv) == torch.Tensor:
            t_inv = t_inv

        return 1/min(t_inv, 1e-2)

    
    @staticmethod
    def safe_log_one_plus_exp(x: torch.Tensor, threshold: float = 5):
        output = torch.zeros_like(x)
        output[x < threshold] = torch.log(1 + torch.exp(x[x < threshold]))
        output[x >= threshold] = x[x >= threshold]
        return output
    
    @staticmethod
    def safe_kl_div(base_probabilities: torch.Tensor, altered_probabilities: torch.Tensor):

        output = torch.zeros_like(base_probabilities)
        suitable_indices = (base_probabilities > 1e-3) & (altered_probabilities > 1e-3)
        output[suitable_indices] = altered_probabilities[suitable_indices] * torch.log(altered_probabilities[suitable_indices] / base_probabilities[suitable_indices])

        return output.sum(dim=-1)


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