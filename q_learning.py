import torch
from torch.optim import AdamW

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
        self.state_buffer = StateBuffer(self.config["buffer_size"])
        self.gridworld_config = gridworld_config
        self.config["gamma"] = (1-3/(self.config["buffer_size"]))
        self.config["buffer_size"] = self.config["buffer_size"]

        self.q_agent = QAgent(reward_cap = self.config["reward_cap"], gamma = self.config["gamma"], **q_agent_config)

        self.optimizer = AdamW(self.q_agent.network.parameters(), **adamw_config)

        self.beta = 1.0
    
    def train(
            self,
            steps: int):
        
        gridworld = TomatoGrid(**self.gridworld_config)

        outputs = []

        for _ in range(steps):
            dict_ = {}
            
            state = gridworld.get_state_tensor()
            dict_["state"] = state

            valid_actions = gridworld.get_valid_actions()
            action_validity = torch.tensor([action in valid_actions for action in list(Action)])
            dict_["action_validity"] = action_validity

            action_idx = self.q_agent.get_action(state.unsqueeze(0), action_validity=action_validity)
            dict_["action"] = torch.tensor(action_idx)


            action = list(Action)[action_idx]
            output = gridworld.update_grid(action)
            dict_["reward"] = torch.tensor(output.misspecified_reward)/13
            self.state_buffer.add(dict_)
        
            if len(self.state_buffer) == self.config["buffer_size"]:
                loss_output = self.q_agent.get_loss(self.state_buffer.get_batch(), beta=self.beta)

                self.optimizer.zero_grad()
                loss = loss_output["loss"]
                loss.backward()
                self.optimizer.step()

            outputs.append(output)

            print(output.misspecified_reward)

            if self.beta < 10:
                self.beta += 0.1


if __name__ == "__main__":
    config = {
        "buffer_size": 100,
        "reward_cap": 1,
    }
    adamw_config = {
        "lr": 1e-2,
        "weight_decay": 1e-5
    }
    q_learning = QLearning(config=config, adamw_config=adamw_config)
    q_learning.train(steps=10000)