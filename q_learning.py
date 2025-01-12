import torch
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
    
    def train(
            self,
            steps: int):
        
        gridworld = TomatoGrid(**self.gridworld_config)

        for step_idx in range(steps):
            dict_ = {}
            
            state = gridworld.get_state_tensor()
            dict_["state"] = state

            valid_actions = gridworld.get_valid_actions()
            action_validity = torch.tensor([action in valid_actions for action in list(Action)])
            dict_["action_validity"] = action_validity

            action_idx = self.q_agent.get_action(state = state.unsqueeze(0), action_validity=action_validity)
            dict_["action"] = torch.tensor(action_idx)


            action = list(Action)[action_idx]
            output = gridworld.update_grid(action)

            dict_["next_state"] = gridworld.get_state_tensor()

            dict_["reward"] = torch.tensor(output.misspecified_reward)/13
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


if __name__ == "__main__":
    config = {
        "buffer_size": 100000,
        "batch_size": 1024,
        "reward_cap": 1,
    }
    adamw_config = {
        "lr": 1e-2,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": 0.99,
        "reward_cap": 1,
        "beta_sample": 1,
        "beta_train": None
    }
    q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
    q_learning.train(steps=100000)