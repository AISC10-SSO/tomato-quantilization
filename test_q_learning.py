from utils import device, sample_random_policy
from utils.learning import QLearning
import numpy as np
import torch
import pandas as pd
from itertools import chain, product
from typing import Literal
import os
from tqdm import tqdm

def main():
    torch.set_default_device(device)

    run_test(
        repeats=3, 
        save_path="Q Learning/Data/only_boltzmann_sampling.csv",
        fixed_kwargs={
            "gamma": 0.99,
            "steps": 100_000,
            "t_inv_sample": 1/13
        },
        variable_kwargs={
            "t_inv_train_deploy": [0.1/13, 0.2/13, 0.5/13, 1/13, 1.5/13, 2/13, 3/13, 5/13],
        }
    )

    run_test(
        repeats=3, 
        save_path="Q Learning/Data/soft_q_learning.csv",
        fixed_kwargs={
            "gamma": 0.99,
            "steps": 100_000,
            "t_inv_sample": 1/13,
            "kl_divergence_coefficient": "auto"
        },
        variable_kwargs={
            "t_inv_train_deploy": [0.1/13, 0.2/13, 0.5/13, 1/13, 1.5/13, 2/13, 3/13, 5/13],
        }
    )

    generate_quantile_data(
        path="Q Learning/Data/quantile_data_misspecified_reward_13.csv",
        misspecified_reward_value=13,
        timesteps=100,
        runs=10_000
    )
    generate_quantile_data(
        path="Q Learning/Data/quantile_data_misspecified_reward_20.csv",
        misspecified_reward_value=20,
        timesteps=100,
        runs=10_000
    )

    quantiles = [0.5, 0.1, 0.02, 0.005, 0.001]
    kl_divergence_targets = [np.log(1/quantile)/100 for quantile in quantiles]

    for misspecified_reward_value in [13, 20]:
        quantile_data = pd.read_csv(f"Q Learning/Data/quantile_data_misspecified_reward_{misspecified_reward_value}.csv")
        quantile_data = quantile_data.sort_values(by="misspecified_reward", ascending=False).reset_index(drop=True)

        misspecified_reward_thresholds = [quantile_data["misspecified_reward"][int(len(quantile_data) * quantile)] for quantile in quantiles]

        run_test(
            repeats=3, 
            save_path=f"Q Learning/Data/soft_q_learning_q_cap_misspecified_reward_{misspecified_reward_value}.csv",
            fixed_kwargs={
                "gamma": 0.99,
                "steps": 100_000,
                "t_inv_sample": 1/13,
                "t_inv_train_deploy": 1/13,
                "kl_divergence_coefficient": "auto",
                "misspecified_reward_value": misspecified_reward_value
            },
            variable_kwargs={
                "q_cap": misspecified_reward_thresholds,
                "kl_divergence_target": kl_divergence_targets
            },
            variable_kwarg_gather_type = "zip"
        )

    run_test(
        repeats=3, 
        save_path="Q Learning/Data/soft_q_learning_misspecified_reward_20.csv",
        fixed_kwargs={
            "gamma": 0.99,
            "steps": 100_000,
            "t_inv_sample": 1/13,
            "kl_divergence_coefficient": "auto",
            "misspecified_reward_value": 20
        },
        variable_kwargs={
            "t_inv_train_deploy": [0.1/13, 0.2/13, 0.5/13, 1/13, 1.5/13, 2/13, 3/13, 5/13],
        }
    )

def generate_quantile_data(path: str, misspecified_reward_value: float, timesteps: int, runs: int):
    if os.path.exists(path):
        print(f"File {path} already exists. Skipping...")
        return
    
    print(f"Generating quantile data for misspecified reward value {misspecified_reward_value}...")

    dfs = []

    for _ in tqdm(range(runs)):
        true_utility, misspecified_reward, __ = sample_random_policy(steps=timesteps, misspecified_reward_value=misspecified_reward_value)
        dfs.append(pd.DataFrame({"true_utility": [true_utility], "misspecified_reward": [misspecified_reward]}))

    final_df = pd.concat(dfs)
    final_df.to_csv(path)

    return


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
        print(kwarg_product)
        kwargs = {key: value for key, value in zip(variable_kwargs.keys(), kwarg_product)}
        for key, value in fixed_kwargs.items():
            kwargs[key] = value

        df_out, config_dict = test_q_learning(runs=repeats, **kwargs)

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
        t_inv_sample: float|None = None,
        gamma: float|None = None,
        reward_cap: float|None = None,
        q_cap: float|None = None,
        runs: int = 10,
        steps: int = 100_000,
        kl_divergence_coefficient: float|None = None,
        kl_divergence_target: float|None = None,
        misspecified_reward_value: float = 13
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
    gridworld_config = {
        "misspecified_reward_value": misspecified_reward_value
    }

    outputs = []
    for _ in range(runs):
        q_learning = QLearning(config=config, adamw_config=adamw_config, q_agent_config=q_agent_config, gridworld_config=gridworld_config)
        q_learning.train(steps=steps)

        outputs.append(q_learning.outputs[-1])

    output_df = pd.DataFrame(outputs)

    config_dict = dict(chain(config.items(), q_agent_config.items(), gridworld_config.items()))

    return output_df, config_dict

if __name__ == "__main__":
    main()