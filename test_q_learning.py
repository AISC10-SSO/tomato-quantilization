from utils import device
from utils.learning import QLearning
import numpy as np
import torch
import pandas as pd

def main():
    torch.set_default_device(device)

    test_q_learning(
        t_inv_train_deploy=1/13,
        t_inv_sample=1/13,
        gamma=0.99,
        runs=1,
        q_cap=0.61*13,
        kl_divergence_coefficient="auto",
        kl_divergence_target=np.log(100)/100,
        steps=100_000)

def test_q_learning(
        t_inv_train_deploy: float|None = None,
        t_inv_sample: float|None = None,
        gamma: float|None = None,
        reward_cap: float|None = None,
        q_cap: float|None = None,
        runs: int = 10,
        steps: int = 100_000,
        kl_divergence_coefficient: float|None = None,
        kl_divergence_target: float|None = None
):

    config = {
        "buffer_size": 1_000_000,
        "batch_size": 1024,
        "kl_divergence_target": kl_divergence_target
    }
    adamw_config = {
        "lr": 2e-3,
        "weight_decay": 1e-5
    }
    q_agent_config = {
        "gamma": gamma,
        "reward_cap": reward_cap,
        "t_inv_sample": t_inv_sample,
        "t_inv_deploy": t_inv_train_deploy,
        "kl_divergence_coefficient": kl_divergence_coefficient,
        "variable_t_inv": config["kl_divergence_target"] is not None,
        "q_cap": q_cap
    }

    outputs = []
    for _ in range(runs):
        q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config)
        q_learning.train(steps=steps)

        outputs.extend(q_learning.outputs)

    output_df = pd.DataFrame(outputs)
    output_df["t_inv_train_deploy"] = t_inv_train_deploy
    output_df["t_inv_sample"] = t_inv_sample
    output_df["gamma"] = gamma
    output_df["reward_cap"] = reward_cap

    return output_df

if __name__ == "__main__":
    main()