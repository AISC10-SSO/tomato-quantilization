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

        self.grid_size = 6*9
        self.input_channels = input_channels
        self.d_res = 32

        self.W_in = nn.Linear(self.grid_size * input_channels, self.d_res)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_res, self.d_res * 4),
                nn.GELU(),
                nn.Linear(self.d_res * 4, self.d_res)
            )
            for _ in range(3)
        ])
        
        self.W_out = nn.Linear(self.d_res, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q Network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size)

        """
        x = F.relu(self.W_in(x.flatten(start_dim=1)))
        for layer in self.layers:
            x =  x + F.relu(layer(x))
        return self.W_out(x)
    
class QAgent:
    def __init__(self, input_channels: int = 6, action_size: int = 5, gamma: float = 0.99, reward_cap: float = 1):

        self.input_channels = input_channels
        self.action_size = action_size

        self.network = QNetwork(input_channels, action_size)

        self.gamma = gamma

        self.q_cap = (1/(1-gamma)) * reward_cap


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

        outputs_capped = outputs.clamp(max=self.q_cap)

        probabilities = F.softmax(outputs * beta, dim=1)
        predicted_values = torch.einsum(
            "ba,ba->b",
            probabilities,
            outputs_capped)[1:].detach()
        
        # Now get the predicted rewards for the actions that were actually taken
        # Then compare to the rewards that were received for that action, plus predicted future rewards, times gamma
        predicted_rewards = outputs[torch.arange(outputs.size(0)), data["action"]][1:]

        target_rewards = data["reward"][:-1] + self.gamma * predicted_values

        # Calculate kl divergence between probabilities and uniform distribution
        # This will be used to restrict the beta value in future
        kl_divergence = F.kl_div(
            probabilities,
            F.softmax(invalid_actions_mask, dim=1),
            reduction="batchmean")
    
    

        loss = F.mse_loss(predicted_rewards, target_rewards)
        return {"loss": loss,
                "outputs": outputs,
                "probabilities": probabilities,
                "kl_divergence": kl_divergence}
    
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