from utils import device
from utils.learning import QLearning

if __name__ == "__main__":
    config = {
        "buffer_size": 1_000_000,
        "batch_size": 1024,
    }
    adamw_config = {
        "lr": 1e-3,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": 0.99,
        "reward_cap": None,
        "beta_sample": 2/13,
        "beta_train": 10/13,
        "beta_deploy": 10/13,
    }
    q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
    q_learning.train(steps=100000)