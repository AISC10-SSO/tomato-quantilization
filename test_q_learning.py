from utils import device
from utils.learning import QLearning
import torch
import pandas as pd

def main():
    torch.set_default_device(device)

    test_q_learning(
        beta_train_deploy=2/13,
        beta_sample=1/13,
        gamma=0.99,
        runs=1,
        reward_cap=7,
        steps=100_000)

def test_q_learning(
        beta_train_deploy: float|None = None,
        beta_sample: float|None = None,
        gamma: float|None = None,
        reward_cap: float|None = None,
        runs: int = 10,
        steps: int = 100_000
):

    config = {
        "buffer_size": 1_000_000,
        "batch_size": 1024,
    }
    adamw_config = {
        "lr": 2e-3,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": gamma,
        "reward_cap": reward_cap,
        "beta_sample": beta_sample,
        "beta_train": beta_train_deploy,
        "beta_deploy": beta_train_deploy,
    }

    outputs = []
    for _ in range(runs):
        q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
        q_learning.train(steps=steps)

        outputs.extend(q_learning.outputs)

    output_df = pd.DataFrame(outputs)
    output_df["beta_train_deploy"] = beta_train_deploy
    output_df["beta_sample"] = beta_sample
    output_df["gamma"] = gamma
    output_df["reward_cap"] = reward_cap

    return output_df

if __name__ == "__main__":
    main()