import torch
import torch.nn.functional as F
from typing import Callable
import utils.functions as UF
from tqdm import tqdm

"""
Contains utilities for running Q-learning with a big matrix rather than a neural network
"""

STATE_DIMENSIONS = (2,)*13 + (29,)

INITIAL_STATE_CODE = [0,0,0,1,1,0,0,0,1,0,1,0,1,10]

"""
X X X X X X X X X   XX XX XX XX XX XX XX XX XX
X X X X X X X O X   XX XX XX XX XX XX XX 00 XX
X T T T T T T _ X   XX 01 02 03 04 05 06 07 XX
X _ _ _ _ _ _ _ X   XX 08 09 10 11 12 13 14 XX
X _ _ _ _ _ _ _ X   XX 15 16 17 18 19 20 21 XX
X T T T T T T T X   XX 22 23 24 25 26 27 28 XX
X X X X X X X X X   XX XX XX XX XX XX XX XX XX


Forward map:
    [start_position, action] -> [destination_position]
    [tomato_state] -> [tomato_state_dewatered]
    [destination_position, tomato_state_dewatered] -> [destination_position, tomato_state_watered]
    [tomato_state_watered] -> reward

    Reverse map:
    Reward tensor:
        [*tomato_state, destination_position] -> [*tomato_state, destination_position] reverse the watering
        [*tomato_state] -> [*tomato_state] reverse the drying
        [position] -> [position, action] reverse the action
"""

def create_valid_action_mask() -> torch.Tensor:
    """
    Create a mask of valid actions for each position

    UP DOWN LEFT RIGHT WAIT
    """
    tensor = torch.zeros(29, 5)
    #UP
    tensor[0:7, 0] = -1e9
    #DOWN
    tensor[22:29, 1] = -1e9
    #LEFT
    tensor[1:23:7, 2] = -1e9
    tensor[0, 2] = -1e9
    #RIGHT
    tensor[0:29:7, 3] = -1e9

    return tensor

def roll_dim(tensor: torch.Tensor, function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Roll a function over the first 13 dimensions of a tensor
    """

    for dimension in range(13):
        tensor = tensor.movedim(dimension, -2)
        tensor = function(tensor)
        tensor = tensor.movedim(-2, dimension)
    return tensor


def create_reward_tensor(misspecified_reward_value: int|None):
    tensor = torch.zeros(STATE_DIMENSIONS)

    def function(tensor: torch.Tensor) -> torch.Tensor:
        tensor[..., 0, :] += 1
        return tensor

    roll_dim(tensor, function)

    if misspecified_reward_value is not None:
        tensor[..., 0] = misspecified_reward_value

    return tensor

class TomatoDryingMap():
    def __init__(self, p_unwatered: float = 0.03):
        # Map state [watered, unwatered] to [watered, unwatered]
        # Dimensions: [input_state, output_state]
        # Watered has p_unwatered chance of becoming unwatered
        # Unwatered has 100% chance of staying unwatered
        self.tensor = torch.tensor(
            [[1 - p_unwatered, p_unwatered],
             [0, 1]]
        )

    def backward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map the tomato states to the next state

        Input:
            state_tensor: a tensor of shape [*tomato_state, destination_position]
        """

        # Apply the dewatering map along each of the 13 dimensions representing binary position
        tensor = roll_dim(state_tensor, lambda tensor: torch.einsum("...np,on->...op", tensor, self.tensor))
        return tensor
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map the next state to the tomato states

        Input:
            state_tensor: a tensor of shape [*tomato_state, destination_position]
        """
        tensor = roll_dim(state_tensor, lambda tensor: torch.einsum("...op,on->...np", tensor, self.tensor))
        return tensor

class TomatoWateringMap():
    def __init__(self):
        self.location_to_tomato_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            22: 6,
            23: 7,
            24: 8,
            25: 9,
            26: 10,
            27: 11,
            28: 12,
        }
    
    def backward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map the location, tomato state to the two possible previous tomato states

        Input:
            state_tensor: a tensor of shape [*tomato_state, destination_position]

        Output:
            state_tensor: a tensor of shape [*tomato_state, destination_position, previous_tomato_state]
        """
        # For each position, map the relevant tomato states to the previous tomato states
        for position, tomato_idx in self.location_to_tomato_map.items():
            # The "watered" state is at indices where the binary representation of the position has a 0 in the tomato_idx position
            # The "unwatered" state is at indices where the binary representation of the position has a 1 in the tomato_idx position
            # We need to map the [destination_position=position, tomato_state=watered] to
            # both [destination_position=position, previous_tomato_state=watered] and
            # [destination_position=position, previous_tomato_state=unwatered]
            state_tensor = state_tensor.movedim(tomato_idx, -2)
            state_tensor[..., 1, position] = state_tensor[..., 0, position]
            state_tensor = state_tensor.movedim(-2, tomato_idx)

        return state_tensor
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map the previous tomato state to the current tomato state

        Input:
            state_tensor: a tensor of shape [*tomato_state, destination_position, previous_tomato_state]

        Output:
            state_tensor: a tensor of shape [*tomato_state, destination_position, tomato_state]
        """
        for position, tomato_idx in self.location_to_tomato_map.items():
            state_tensor = state_tensor.movedim(tomato_idx, -2)
            state_tensor[..., 0, position] += state_tensor[..., 1, position]
            state_tensor[..., 1, position] = 0
            state_tensor = state_tensor.movedim(-2, tomato_idx)

        return state_tensor

class PositionActionMap():
    def __init__(self):
        self.tensor = torch.zeros((29, 5, 29))
        # UP DOWN LEFT RIGHT WAIT
        # UP: if square <= 7, then square - 7
        # DOWN: if square >= 22, then square + 7
        # LEFT: if square % 7 != 0, then square - 1
        # RIGHT: if square % 7 != 6, then square + 1
        # WAIT: square
        for square in range(29):
            if square >= 7:
                self.tensor[square, 0, square - 7] = 1
            else:
                self.tensor[square, 0, square] = 1

            if square <= 21:
                self.tensor[square, 1, square + 7] = 1
            else:
                self.tensor[square, 1, square] = 1

            if square % 7 == 1:
                self.tensor[square, 2, square] = 1
            else:
                self.tensor[square, 2, square - 1] = 1

            if square % 7 == 0 or square == 0:
                self.tensor[square, 3, square] = 1
            else:
                self.tensor[square, 3, square + 1] = 1

            self.tensor[square, 4, square] = 1

    def backward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map a probability distribution over position, action to a probability distribution over position

        Input:
            position_action_tensor: a tensor of shape (tomato_state, destination_position)
        Output:
            position_tensor: a tensor of shape (tomato_state, start_position, action)
        """

        # ... = tomato_state, s = start_position, a = action, d = destination_position
        mapped_tensor = torch.einsum("...d,sad->...sa", state_tensor, self.tensor)
        return mapped_tensor
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map a probability distribution over position, action to a probability distribution over position, action
        """

        return torch.einsum("...sa,sad->...d", state_tensor, self.tensor)


class MapCollection():

    def __init__(self, maps: list[TomatoWateringMap|TomatoDryingMap|PositionActionMap]):
        self.maps = maps

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        for map in self.maps:
            state_tensor = map.forward(state_tensor)
        return state_tensor
    
    def backward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        for map in self.maps[::-1]:
            state_tensor = map.backward(state_tensor)
        return state_tensor

class QMatrix():

    def __init__(
            self,*,
            t_inv: float = 1.5/13,
            misspecified_reward_value: int = 13,
            gamma: float = 0.99,
            update_size: float = 0.5,
            kl_divergence_penalty: float|None = None,
            q_cap: float|None = None
            ):
    
        self.tensors = {
            "reward": create_reward_tensor(misspecified_reward_value),
            "utility": create_reward_tensor(None),
        }
        self.q_matrices = {
            key: torch.zeros(STATE_DIMENSIONS + (5,))
            for key in self.tensors.keys()
        }
        self.t_inv = t_inv
        self.valid_action_mask = create_valid_action_mask()
        self.gamma = gamma
        self.update_size = update_size
        self.q_cap = None if q_cap is None else q_cap / (1 - gamma)

        self.kl_divergence_penalty = kl_divergence_penalty

        if kl_divergence_penalty is not None:
            self.q_matrices["kl_divergence"] = torch.zeros(STATE_DIMENSIONS + (5,))


        self.map_collection = MapCollection([
            PositionActionMap(),
            TomatoDryingMap(),
            TomatoWateringMap(),
        ])

    def get_probabilities(self) -> torch.Tensor:

        if self.q_cap is None:
            logits = self.q_matrices["reward"] * self.t_inv + self.valid_action_mask
        else:
            logits = -UF.safe_log_one_plus_exp((self.q_cap - self.q_matrices["reward"]) * self.t_inv) + self.valid_action_mask

        if self.kl_divergence_penalty is not None:
            logits = logits - self.q_matrices["kl_divergence"] * self.kl_divergence_penalty

        return F.softmax(logits, dim=-1)

    def update(self) -> float:

        probabilities = self.get_probabilities()

        target_tensors = {
            key: self.tensors[key] + self.gamma * torch.einsum(
                "...a,...a->...",
                probabilities,
                self.q_matrices[key])
            for key in ["reward", "utility"]
        }

        if self.kl_divergence_penalty is not None:
            base_probabilities = F.softmax(self.valid_action_mask, dim=-1).expand(*STATE_DIMENSIONS, 5)

            kl_divergence = UF.safe_kl_div(base_probabilities, probabilities)
            target_tensors["kl_divergence"] = kl_divergence * self.kl_divergence_penalty + self.gamma * torch.einsum(
                "...a,...a->...",
                probabilities,
                self.q_matrices["kl_divergence"])

        for key in self.q_matrices.keys():
            target_tensors[key] = self.map_collection.backward(target_tensors[key])

        difference_sum = 0
        for key in self.q_matrices.keys():
            difference = target_tensors[key] - self.q_matrices[key]
            self.q_matrices[key] = self.q_matrices[key] + difference * self.update_size
            difference_sum += difference.abs().mean().item()

        return difference_sum
    
    def train(self, max_timesteps: int = 10000, min_difference: float|None = 1e-3):
        for i in (progress_bar := tqdm(range(max_timesteps))): # Walrus
            difference_sum = self.update()
            progress_bar.set_description(f"Difference sum: {difference_sum:.3g}")
            if min_difference is not None and difference_sum < min_difference:
                print(f"Converged after {i} iterations, difference sum: {difference_sum}")
                break

    
    def get_reward_and_utility(self, timesteps: int = 100) -> dict[str, float]:

        state = torch.zeros(STATE_DIMENSIONS)
        state[*INITIAL_STATE_CODE] = 1

        action_tensor = self.get_probabilities()

        return_dict = {
            "reward": 0,
            "utility": 0,
        }

        for _ in range(timesteps):
            action_probabilities = torch.einsum("...a,...->...a", action_tensor, state)
            state = self.map_collection.forward(action_probabilities)

            for key in return_dict.keys():
                return_dict[key] += (state * self.tensors[key]).sum().item()

        return return_dict
