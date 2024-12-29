from utils import TomatoGrid, Action, iterative_complexity_reduction, InvalidActionSetting
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import random

def main():

    """
    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/no_iterations.csv",
        iterations=0)

    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/1_iteration.csv",
        iterations=1)

    save_policy_datapoints(
        datapoints_count=1_000_000,
        save_path="datapoints/1M_datapoints.csv",
        iterations=0)

    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/no_iterations_random_move_if_invalid.csv",
        iterations=0,
        invalid_action_setting=InvalidActionSetting.RANDOM)
    
    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/1_iteration_random_move_if_invalid.csv",
        iterations=1,
        invalid_action_setting=InvalidActionSetting.RANDOM)
    """

    plot_policy_datapoints(save_path="plots", data_path="datapoints")

def plot_policy_datapoints(save_path: str, data_path: str):

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            title = file[:-len(".csv")]
            plot_single_policy_datapoints(
                read_path=f"{data_path}/{file}",
                save_path=f"{save_path}/",
                title=title)


def plot_single_policy_datapoints(
        read_path: str,
        save_path: str,
        title: str):

    df = pd.read_csv(read_path)
    plt.scatter(df["misspecified_reward"], df["true_utility"], alpha=0.1)
    plt.xlabel("Misspecified Reward")
    plt.ylabel("True Utility")
    plt.title(title)
    plt.savefig(f"{save_path}/{title}_scatter.png")

    plt.clf()

    # 10 datapoint quantile
    ten_datapoint_quantile_log = int(np.log10(len(df)/10))

    quantiles = [1] + [
        x * 10**(-i)
        for i in range(1,ten_datapoint_quantile_log)
        for x in [0.5, 0.2, 0.1]
    ]

    quantile_true_utilities = []
    quantile_false_utilities = []

    df_sorted = df.sort_values(by="misspecified_reward", ascending=False)

    for q in quantiles:
        quantile_true_utilities.append(df_sorted["true_utility"].iloc[:int(q * len(df))].mean())
        quantile_false_utilities.append(df_sorted["misspecified_reward"].iloc[:int(q * len(df))].mean())

    plt.plot(quantiles, quantile_true_utilities, label="True Utility")
    plt.plot(quantiles, quantile_false_utilities, label="Misspecified Reward")
    plt.xlabel("Quantile")
    plt.ylabel("Utility")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}/{title}_quantiles.png")
    plt.clf()

    df["misspecified_reward_normalized"] = ((df["misspecified_reward"] - np.mean(df["misspecified_reward"])) / np.std(df["misspecified_reward"]))

    b_values = [
        x * 0.25 for x in range(1, 8 + ten_datapoint_quantile_log*2)
    ]

    b_true_utilities = []
    b_false_utilities = []
    sqrt_kl_divergences = []

    for b in b_values:
        df["weights"] = np.exp(b * df["misspecified_reward_normalized"])
        df["weights"] = df["weights"] / np.sum(df["weights"])
        df["weighted_true_utility"] = df["true_utility"] * df["weights"] 
        df["weighted_misspecified_reward"] = df["misspecified_reward"] * df["weights"]

        b_true_utilities.append(df["weighted_true_utility"].sum())
        b_false_utilities.append(df["weighted_misspecified_reward"].sum())

        kl_divergence = np.sum(df["weights"] * (np.log(df["weights"]))) - np.log(1/len(df))
        sqrt_kl_divergences.append(np.sqrt(kl_divergence))

    plt.plot(sqrt_kl_divergences, b_true_utilities, label="True Utility")
    plt.plot(sqrt_kl_divergences, b_false_utilities, label="Misspecified Reward")
    plt.xlabel("sqrt(KL Divergence)")
    plt.ylabel("Utility")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}/{title}_kl_divergence.png")
    plt.clf()

    return quantile_true_utilities[-1], quantile_false_utilities[-1]

def save_policy_datapoints(
        datapoints_count: int = 10000,
        save_path: str = "datapoints.csv",
        iterations: int = 0,
        invalid_action_setting: InvalidActionSetting = InvalidActionSetting.WAIT):

    datapoints = []
    for _ in tqdm(range(datapoints_count)):
        true_utility, misspecified_reward, complexity = sample_random_policy(iterations=iterations, invalid_action_setting=invalid_action_setting)
        datapoints.append((true_utility, misspecified_reward, complexity))

    df = pd.DataFrame(datapoints, columns=["true_utility", "misspecified_reward", "complexity"])
    df.to_csv(save_path, index=False)

def sample_random_policy(steps: int = 1000, iterations: int = 0, invalid_action_setting: InvalidActionSetting = InvalidActionSetting.WAIT):
    grid = TomatoGrid(invalid_action_setting=invalid_action_setting)

    total_true_utility = 0
    total_misspecified_reward = 0

    if iterations > 0:
        sequence, complexity = iterative_complexity_reduction(length=steps, iterations=iterations)
        for action in sequence:
            grid.update_grid(Action(action))
            total_true_utility += grid.get_current_utility()[0]
            total_misspecified_reward += grid.get_current_utility()[1]
    else:
        complexity = 1
        for _ in range(steps):
            action = random.choice(list(Action))
            grid.update_grid(action)
            total_true_utility += grid.get_current_utility()[0]
            total_misspecified_reward += grid.get_current_utility()[1]

    average_true_utility = total_true_utility / steps
    average_misspecified_reward = total_misspecified_reward / steps

    return average_true_utility, average_misspecified_reward, complexity


if __name__ == "__main__":
    main()
