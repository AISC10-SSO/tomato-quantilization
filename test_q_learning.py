from utils import device, sample_random_policy
from utils.learning import QLearning
import numpy as np
import torch
import pandas as pd
from itertools import chain, product
from typing import Literal
import os

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    torch.set_default_device(device)

    run_test(
        repeats=1,
        save_path="Q Learning/Data/boltzmann_sampling.csv",
        fixed_kwargs={
            "t_inv_train_deploy":  2/13,
        },
        variable_kwargs={
        },
    )


def run_test(
        *,
        repeats: int,
        save_path: str,
        fixed_kwargs: dict,
        variable_kwargs: dict,
        variable_kwarg_gather_type: Literal["zip", "product"] = "product") -> None:

    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping...")
        return
    
    print(f"Running test and saving to {save_path}...")

    match variable_kwarg_gather_type:
        case "zip":
            variable_kwargs_gathered = zip(*variable_kwargs.values())
        case "product":
            variable_kwargs_gathered = product(*variable_kwargs.values())

    final_dfs = []
    for kwarg_product in variable_kwargs_gathered:
        kwargs = {key: value for key, value in zip(variable_kwargs.keys(), kwarg_product)}
        for key, value in fixed_kwargs.items():
            kwargs[key] = value

        df_out, config_dict = test_q_learning(**kwargs)

        processed_df = process_df(df_out, config_dict)

        final_dfs.append(processed_df)

    final_df = pd.concat(final_dfs)
    final_df.to_csv(save_path)

def process_df(df_out, config_dict):
    new_df = pd.DataFrame()
    for column in df_out.columns:
        new_df[f"{column}_mean"] = [df_out[column].mean()]
        new_df[f"{column}_stderr"] = [df_out[column].std() / np.sqrt(len(df_out))]

    for key, value in config_dict.items():
        new_df[key] = value

    return new_df


def test_q_learning(
        t_inv_train_deploy: float|None = None,
        t_inv_sample: float|None|Literal["auto"] = "auto",
        gamma: float|None = 0.99,
        q_cap: float|None = None,
        steps: int = 100_000,
        kl_divergence_coefficient: float|None = None,
        misspecified_reward_value: float = 13
):

    config = {
        "buffer_size": 100_000,
        "batch_size": 1024,
    }
    adamw_config = {
        "lr": 2e-3,
        "weight_decay": 1e-2
    }
    q_agent_config = {
        "gamma": gamma,
        "t_inv_sample": t_inv_sample,
        "t_inv_deploy": t_inv_train_deploy,
        "kl_divergence_coefficient": kl_divergence_coefficient,
        "q_cap": q_cap
    }
    gridworld_config = {
        "misspecified_reward_value": misspecified_reward_value
    }



    q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config, gridworld_config=gridworld_config)
    q_learning.train(steps=steps)

    output_df = pd.DataFrame(q_learning.outputs)

    sns.lineplot(data=output_df, x="step", y="misspecified_reward")
    sns.lineplot(data=output_df, x="step", y="true_utility")
    plt.show()
    plt.clf()

    config_dict = dict(chain(config.items(), q_agent_config.items(), gridworld_config.items()))

    return output_df, config_dict

if __name__ == "__main__":
    main()