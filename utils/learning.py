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
    def __init__(
            self, *,
            input_channels: int = 6,
            action_size: int = 5,
            gamma: float = 0.99,
            beta_train: float|None = None,
            beta_sample: float|None = None,
            reward_cap: float|None = None):

        self.input_channels = input_channels
        self.action_size = action_size

        self.network = QNetwork(input_channels, action_size)

        self.gamma = gamma
        self.beta_train = beta_train
        self.beta_sample = beta_sample

        self.q_cap = None if reward_cap is None else (1/(1-gamma)) * reward_cap

        # Add target network for stable learning
        self.target_network = QNetwork(input_channels, action_size)
        self.target_network.load_state_dict(self.network.state_dict())

    def get_loss(self, data: dict[str, torch.Tensor]):
        """
        Get the loss from a batch of data, containing rewards per state, actions taken, and input states

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing "state", "next_state", "reward", "action", and "action_validity"
            beta (float): Beta value for the temperature scaling

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "loss", "outputs", "probabilities", and "kl_divergence"
        """

        # Predict the rewards for each state, given the beta value
        # This means taking the probabilities of each action and multiplying them by the predicted rewards (Q values)
        # Throw away the first state as we don't know what came before it
        invalid_actions_mask = (~data["action_validity"]).float() * -1e9
        outputs = self.network(data["state"]) + invalid_actions_mask

        if self.beta_sample is not None:
            probabilities = F.softmax(outputs * self.beta_sample, dim=1)
        else:
            probabilities = torch.max(outputs, dim=-1)
        
        # Use target network with temperature scaling
        with torch.no_grad():
            next_q_values = self.target_network(data["next_state"]) + invalid_actions_mask
            next_q_values_capped = next_q_values.clamp(max=self.q_cap)

            next_probabilities = self.beta_softmax(
                next_q_values,
                beta=self.beta_train,
                action_validity=data["action_validity"])
            
            next_values = torch.einsum("ba,ba->b", next_probabilities, next_q_values_capped)

            target_rewards = data["reward"] + self.gamma * next_values

        predicted_rewards = outputs.gather(1, data["action"].unsqueeze(1)).squeeze()
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
    
    def get_action(
            self, *,
            state: torch.Tensor,
            action_validity: torch.Tensor|None = None) -> int:
        """
        Get the action to take from the network

        Args:
            state (torch.Tensor): Input state of shape (input_channels, height, width)
            action_validity (torch.Tensor|None): Action validity mask of shape (action_size)

        Returns:
            int: Action to take
        """
        outputs = self.network(state)
        probabilities = self.beta_softmax(outputs, beta=self.beta_sample, action_validity=action_validity)
        # Choose randomly
        return torch.multinomial(probabilities, 1).item()
        
    @classmethod
    def beta_softmax(cls, outputs: torch.Tensor, beta: float|None = None, action_validity: torch.Tensor|None = None):
        """
        Softmax with beta scaling

        Args:
            outputs (torch.Tensor): Outputs from the network of shape (batch_size, action_size)
            beta (float|None): Beta value for the temperature scaling
            action_validity (torch.Tensor|None): Action validity mask of shape (batch_size, action_size)

        Returns:
            torch.Tensor: Probabilities of shape (batch_size, action_size)
        """
        if beta is not None:
            probabilities = F.softmax(outputs * beta, dim=-1)
        else:
            probabilities = torch.max(outputs, dim=-1)

        if action_validity is not None:
            probabilities = probabilities + (~action_validity).float() * -1e9

        return probabilities

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