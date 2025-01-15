from utils import device
from utils.learning import QLearning
import torch

if __name__ == "__main__":
    torch.set_default_device(device)

    config = {
        "buffer_size": 1_000_000,
        "batch_size": 1024,
    }
    adamw_config = {
        "lr": 2e-3,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": 0.99,
        "reward_cap": None,
        "beta_sample": 1/13,
        "beta_train": 1/13,
        "beta_deploy": 1/13,
    }
    q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
    q_learning.train(steps=100000)