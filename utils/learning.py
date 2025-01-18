import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        return self.fc2(x)
    
class QAgent:
    def __init__(
            self, *,
            input_channels: int = 6,
            action_size: int = 5,
            gamma: float = 1.99,
            kl_divergence_coefficient: float|None = None,
            t_inv_sample: float|None = None,
            t_inv_deploy: float|None = None,
            reward_cap: float|None = None,
            variable_t_inv: bool = False):

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

        # self.q_cap = None if reward_cap is None else (1/(1-gamma)) * reward_cap
        self.q_cap = None

        # Create two target networks
        self.target_network_1 = QNetwork(input_channels, action_size)
        self.target_network_2 = QNetwork(input_channels, action_size)
        
        self.target_network_1.load_state_dict(self.network_1.state_dict())
        self.target_network_2.load_state_dict(self.network_2.state_dict())

        self.kl_divergence_coefficient = kl_divergence_coefficient

        self.average_q_value = 0

    def get_loss(self, data: dict[str, torch.Tensor]):
        """
        Get the loss from a batch of data, containing rewards per state, actions taken, and input states

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing "state", "next_state", "reward", "action", and "action_validity"
            t_inv (float): inverse T value for the temperature scaling

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "loss", "outputs", "probabilities", and "kl_divergence"
        """

        # Predict the rewards for each state, given the beta value
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

            next_values_1 = torch.einsum("ba,ba->b", next_probabilities_1, next_q_values_1)
            next_values_2 = torch.einsum("ba,ba->b", next_probabilities_2, next_q_values_2)

            reward = data["reward"]
            if self.reward_cap is not None:
                reward = reward.clamp(max=self.reward_cap)

            target_rewards_1 = reward + self.gamma * next_values_1
            target_rewards_2 = reward + self.gamma * next_values_2

            if self.kl_divergence_coefficient is not None:
                next_state_prior = F.softmax(next_state_invalid_actions_mask, dim=-1)

                next_kl_divergence_1 = F.kl_div(torch.log(next_probabilities_1), next_state_prior, reduction="none").sum(dim=-1)
                next_kl_divergence_2 = F.kl_div(torch.log(next_probabilities_2), next_state_prior, reduction="none").sum(dim=-1)

                target_rewards_1 = target_rewards_1 - self.kl_divergence_coefficient * next_kl_divergence_1
                target_rewards_2 = target_rewards_2 - self.kl_divergence_coefficient * next_kl_divergence_2

        # Calculate losses for both networks
        outputs_1 = self.network_1(data["state"])
        outputs_2 = self.network_2(data["state"])

        predicted_rewards_1 = outputs_1.gather(1, data["action"].unsqueeze(1)).squeeze()
        predicted_rewards_2 = outputs_2.gather(1, data["action"].unsqueeze(1)).squeeze()

        # Swap the target rewards for the two networks
        loss_1 = F.smooth_l1_loss(predicted_rewards_1, target_rewards_2 - self.average_q_value)
        loss_2 = F.smooth_l1_loss(predicted_rewards_2, target_rewards_1 - self.average_q_value)

        self.average_q_value = 0.9 * self.average_q_value + 0.1 * data["reward"].float().mean().item()

        # Total loss is the sum of both networks' losses
        loss = loss_1 + loss_2

        # Use average of both networks for probabilities and outputs
        outputs = (outputs_1 + outputs_2) / 2
        probabilities = self.get_probabilities(outputs.detach(), mode="sample", action_validity=data["action_validity"])


        kl_divergence = F.kl_div(torch.log(probabilities), F.softmax(invalid_actions_mask, dim=-1), reduction="none").sum(dim=-1)

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
            self.average_q_value for the average q value
        Returns:
            torch.Tensor: Probabilities of shape (batch_size, action_size)
        """

        # Mask out invalid actions
        if mode == "sample":
            t_inv = self.t_inv_sample
        elif mode == "deploy":
            t_inv = self.t_inv_deploy

        # Subtract the maximum value from each row to prevent overflow
        outputs = outputs - outputs.max(dim=-1, keepdim=True).values

        if action_validity is None:
            action_validity_mask = torch.zeros_like(outputs)
        else:
            action_validity_mask = (~action_validity).float() * -1e9

        # Apply temperature scaling if t_inv is provided
        if t_inv is None:
            # If no t_inv, use argmax
            probabilities = torch.zeros_like(outputs)
            probabilities[torch.arange(outputs.shape[0]), torch.argmax(outputs + action_validity_mask, dim=-1)] = 1
        else:
            if self.q_cap is None:
                # If no q_cap, use softmax
                probabilities = F.softmax(outputs * t_inv + action_validity_mask, dim=-1)
            else:
                # If q_cap is provided, use capped softmax
                log_probabilities = -self.safe_log_one_plus_exp(t_inv * (self.q_cap - (outputs + self.average_q_value))) + action_validity_mask
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
    
    @staticmethod
    def safe_log_one_plus_exp(x: torch.Tensor, threshold: float = 5):
        output = torch.zeros_like(x)
        output[x < threshold] = torch.log(1 + torch.exp(x[x < threshold]))
        output[x >= threshold] = x[x >= threshold]
        return output

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

        self.optimizer_1 = torch.optim.AdamW(self.q_agent.network_1.parameters(), **adamw_config)
        self.optimizer_2 = torch.optim.AdamW(self.q_agent.network_2.parameters(), **adamw_config)

        self.outputs = []

    def get_action_validity(self, gridworld: TomatoGrid):
        valid_actions = gridworld.get_valid_actions()
        action_validity = torch.tensor([action in valid_actions for action in list(Action)])
        return action_validity
    
    def train(
            self,
            steps: int):
        
        gridworld = TomatoGrid(**self.gridworld_config)

        for step_idx in range(steps):
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

                    self.optimizer_1.zero_grad()
                    self.optimizer_2.zero_grad()
                    loss = loss_output["loss"]

                    if self.config["kl_divergence_target"] is not None:
                        kl_divergence = loss_output["kl_divergence"]
                        kl_divergence_loss = F.smooth_l1_loss(
                            kl_divergence / self.config["kl_divergence_target"],
                            torch.ones_like(kl_divergence))
                        loss += kl_divergence_loss

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.q_agent.network_1.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(self.q_agent.network_2.parameters(), max_norm=1.0)

                    self.optimizer_1.step()
                    self.optimizer_2.step()
                
                    self.q_agent.update_target_network(tau=0.01)

            if step_idx % 1000 == 0:
                test_output = self.test_model()
                print(test_output)
                self.outputs.append(test_output)

    def test_model(self):
        gridworlds = [TomatoGrid(**self.gridworld_config) for _ in range(10)]

        outputs = []

        for _ in range(100):
            state = torch.stack([gridworld.get_state_tensor() for gridworld in gridworlds])
            action_indices = self.q_agent.get_action(state = state, action_validity=None, mode="deploy")
            actions = [list(Action)[action_idx] for action_idx in action_indices]
            outputs += [gridworld.update_grid(action) for gridworld, action in zip(gridworlds, actions)]

        misspecified_reward = np.mean([output.misspecified_reward for output in outputs])
        true_utility = np.mean([output.true_utility for output in outputs])

        return {"misspecified_reward": misspecified_reward, "true_utility": true_utility}
