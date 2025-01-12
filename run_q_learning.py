import torch
import numpy as np
from torch.optim import AdamW
from enum import Enum

from utils.learning import QAgent, StateBuffer
from utils import TomatoGrid, Action

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

        self.optimizer = AdamW(self.q_agent.network.parameters(), **adamw_config)

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

            dict_["action_validity"] = self.get_action_validity(gridworld)

            action_idx = self.q_agent.get_action(state = state.unsqueeze(0), action_validity=dict_["action_validity"], mode="sample")
            dict_["action"] = torch.tensor(action_idx)


            action = list(Action)[action_idx]
            output = gridworld.update_grid(action)

            dict_["next_state"] = gridworld.get_state_tensor()
            dict_["next_state_action_validity"] = self.get_action_validity(gridworld)

            dict_["reward"] = torch.tensor(output.misspecified_reward)
            self.state_buffer.add(dict_)

            if step_idx % 100 == 0 and step_idx > 0:
                try:
                    loss_output = self.q_agent.get_loss(self.state_buffer.get_batch())

                    self.optimizer.zero_grad()
                    loss = loss_output["loss"]
                    loss.backward()
                    self.optimizer.step()
                
                    self.q_agent.update_target_network(tau=0.01)

                except ValueError:
                    pass

            if step_idx % 1000 == 0:
                test_output = self.test_model()
                print(f"Step {step_idx}: Misspecified reward: {test_output['misspecified_reward']}, True utility: {test_output['true_utility']}")

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


if __name__ == "__main__":
    config = {
        "buffer_size": 100000,
        "batch_size": 1024,
    }
    adamw_config = {
        "lr": 1e-2,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": 0.99,
        "reward_cap": None,
        "beta_sample": 0.5,
        "beta_train": 2,
        "beta_deploy": 2,
    }
    q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
    q_learning.train(steps=100000)