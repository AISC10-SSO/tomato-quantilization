import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

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
    def __init__(self, input_channels: int = 6, action_size: int = 5, gamma: float = 0.99, reward_cap: float = 1):

        self.input_channels = input_channels
        self.action_size = action_size

        self.network = QNetwork(input_channels, action_size)

        self.gamma = gamma

        self.q_cap = (1/(1-gamma)) * reward_cap

        # Add target network for stable learning
        self.target_network = QNetwork(input_channels, action_size)
        self.target_network.load_state_dict(self.network.state_dict())

    def get_loss(self, data: dict[str, torch.Tensor], beta: float):
        """
        Get the loss from a batch of data, containing rewards per state, actions taken, and input states

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing "input_states", "rewards", and "actions"
            beta (float): Beta value for the temperature scaling

        Returns:
            torch.Tensor: Loss value
        """

        # Predict the rewards for each state, given the beta value
        # This means taking the probabilities of each action and multiplying them by the predicted rewards (Q values)
        # Throw away the first state as we don't know what came before it
        invalid_actions_mask = (~data["action_validity"]).float() * -1e9
        outputs = self.network(data["state"]) + invalid_actions_mask

        # outputs_capped = outputs.clamp(max=self.q_cap)

        probabilities = F.softmax(outputs * beta, dim=1)
        
        # Use target network with temperature scaling
        with torch.no_grad():
            next_q_values = self.target_network(data["state"]) + invalid_actions_mask
            next_q_values_capped = next_q_values.clamp(max=self.q_cap)
            next_probs = F.softmax(next_q_values[1:] * beta, dim=1)
            next_values = (next_probs * next_q_values_capped[1:]).sum(dim=1)
            target_rewards = data["reward"][:-1] + self.gamma * next_values

        predicted_rewards = outputs[:-1].gather(1, data["action"][:-1].unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(predicted_rewards, target_rewards)

        # Calculate KL divergence
        base_probabilities = F.softmax(invalid_actions_mask, dim=1)
        probabilities_ratio = base_probabilities / (probabilities + 1e-9) + 1e-9
        kl_divergence = torch.sum(base_probabilities * torch.log(probabilities_ratio), dim=-1).mean()

        return {
            "loss": loss,
            "outputs": outputs,
            "probabilities": probabilities,
            "kl_divergence": kl_divergence
        }
    
    def get_action(self, state: torch.Tensor, action_validity: torch.Tensor|None = None, random: bool = True, beta: float = 1.0) -> int:
        """
        Get the action to take from the network

        Args:
            state (torch.Tensor): Input state of shape (input_channels, height, width)

        Returns:
            int: Action to take
        """
        outputs = self.network(state)
        if random:
            # Get the probabilities of each action
            if action_validity is None:
                probabilities = F.softmax(outputs * beta, dim=-1)
            else:
                probabilities = F.softmax(outputs * beta + (~action_validity).float() * -1e9, dim=-1)
            # Choose randomly
            return torch.multinomial(probabilities, 1).item()
        else:
            return torch.argmax(outputs).item()


class StateBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        self.buffers: dict[str, deque] = {
            "state": deque(maxlen=buffer_size),
            "action": deque(maxlen=buffer_size),
            "reward": deque(maxlen=buffer_size),
            "action_validity": deque(maxlen=buffer_size),
        }

    def add(self, data: dict[str, torch.Tensor]):
        for key, value in data.items():
            self.buffers[key].append(value)

    def get_batch(self) -> dict[str, torch.Tensor]:
        return {k: torch.stack(list(v)) for k, v in self.buffers.items()}
    
    def __len__(self):
        return len(self.buffers["state"])
    
    def clear(self):
        for buffer in self.buffers.values():
            buffer.clear()