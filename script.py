from utils import TomatoGrid, Action, iterative_complexity_reduction, InvalidActionSetting
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import random

def main():

    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/random_prior_wait_if_invalid.csv",
        iterations=0)

    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/simplicity_prior_wait_if_invalid.csv",
        iterations=1)

    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/random_prior_random_move_if_invalid.csv",
        iterations=0,
        invalid_action_setting=InvalidActionSetting.RANDOM)
    
    save_policy_datapoints(
        datapoints_count=10_000,
        save_path="datapoints/simplicity_prior_random_move_if_invalid.csv",
        iterations=1,
        invalid_action_setting=InvalidActionSetting.RANDOM)

    """
    save_policy_datapoints(
        datapoints_count=1_000_000,
        save_path="datapoints/1M_datapoints.csv",
        iterations=0)
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
    
    os.makedirs(os.path.join(save_path, title), exist_ok=True)

    df = pd.read_csv(read_path)

    df["true_utility"] = df["true_utility"] / 13
    df["misspecified_reward"] = df["misspecified_reward"] / 13

    plt.scatter(df["misspecified_reward"], df["true_utility"], alpha=0.1)
    plt.xlabel("Misspecified Reward")
    plt.ylabel("True Utility")
    plt.title(title)
    plt.savefig(f"{save_path}/{title}/reward_scatter.png")

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
    plt.savefig(f"{save_path}/{title}/reward_quantiles.png")
    plt.clf()

    plt.scatter(df["steps_on_tomato"], df["true_utility"], alpha=0.1)
    plt.xlabel("Steps on Tomato")
    plt.ylabel("True Utility")
    plt.title(title)
    plt.savefig(f"{save_path}/{title}/steps_on_tomato_scatter.png")
    plt.clf()

    df_sorted = df.sort_values(by="steps_on_tomato", ascending=False)

    quantile_true_utilities = []
    quantile_steps_on_tomato = []

    for q in quantiles:
        quantile_true_utilities.append(df_sorted["true_utility"].iloc[:int(q * len(df))].mean())
        quantile_steps_on_tomato.append(df_sorted["steps_on_tomato"].iloc[:int(q * len(df))].mean())

    plt.plot(quantiles, quantile_true_utilities, label="True Utility")
    plt.plot(quantiles, quantile_steps_on_tomato, label="Steps on Tomato")
    plt.xlabel("Quantile")
    plt.ylabel("Utility")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}/{title}/steps_on_tomato_quantiles.png")
    plt.clf()

    weight_values = [x * 0.1 for x in range(11)] # relative weighting

    df["misspecified_reward_normalized"] = ((df["misspecified_reward"] - np.mean(df["misspecified_reward"])) / np.std(df["misspecified_reward"]))
    df["steps_on_tomato_normalized"] = ((df["steps_on_tomato"] - np.mean(df["steps_on_tomato"])) / np.std(df["steps_on_tomato"]))

    """
    sqrt_kl_divergences = []
    true_utilities = []
    names = []

    for weight in weight_values:
        sqrt_kl_divergences_for_weight = []
        true_utilities_for_weight = []

        for b in np.arange(0, 100, 0.2):
            df["weights"] = 1

            b_misspecified_reward = b * weight
            b_steps_on_tomato = b * (1 - weight)

            df["weights"] *= np.exp(b_misspecified_reward * df["misspecified_reward_normalized"])
            df["weights"] *= np.exp(b_steps_on_tomato * df["steps_on_tomato_normalized"])

            df["weights"] = df["weights"] / np.sum(df["weights"])

            kl_divergence = np.sum(df["weights"] * np.log(df["weights"])) + np.log(len(df))

            sqrt_kl_divergences_for_weight.append(np.sqrt(kl_divergence))
            true_utilities_for_weight.append(np.sum(df["true_utility"] * df["weights"]))

            if kl_divergence > 10:
                break

        sqrt_kl_divergences.append(sqrt_kl_divergences_for_weight)
        true_utilities.append(true_utilities_for_weight)
        names.append(f"{weight*10:.0f}:{10-weight*10:.0f}")


    for i in range(len(weight_values)):
        plt.plot(sqrt_kl_divergences[i], true_utilities[i], label=names[i], c=plt.cm.viridis(weight_values[i]))
    plt.xlabel("Sqrt KL Divergence")
    plt.ylabel("True Utility")
    plt.legend()
    plt.title(title)
    plt.savefig(f"{save_path}/{title}/sqrt_kl_divergences.png")
    plt.clf()
    """

    true_utilities = []
    names = []

    for w in weight_values:
        df["weighted_mix"] = df["misspecified_reward"] * w + df["steps_on_tomato"] * (1 - w)
        df_sorted = df.sort_values(by="weighted_mix", ascending=False)
        true_utilities_for_weights = []
    
        names.append(f"{w*10:.0f}:{10-w*10:.0f}")

        for q in quantiles:
            true_utilities_for_weights.append(df_sorted["true_utility"].iloc[:int(q * len(df))].mean())

        true_utilities.append(true_utilities_for_weights)

    for i in range(len(weight_values)):
        plt.plot(quantiles, true_utilities[i], c=plt.cm.viridis(weight_values[i]))

    # Create a new figure and axes for the colorbar
    fig = plt.gcf()
    ax = plt.gca()
    
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    # Pass the ax argument to colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Only steps_on_tomato", "Only misspecified_reward"])

    plt.xlabel("Quantile")
    plt.ylabel("True Utility")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.savefig(f"{save_path}/{title}/mixed_quantilization.png", bbox_inches="tight")
    plt.clf()


def save_policy_datapoints(
        datapoints_count: int = 10000,
        save_path: str = "datapoints.csv",
        iterations: int = 0,
        invalid_action_setting: InvalidActionSetting = InvalidActionSetting.WAIT):
    
    if os.path.exists(save_path):
        return

    datapoints = []
    for _ in tqdm(range(datapoints_count)):
        true_utility, misspecified_reward, on_tomato = sample_random_policy(iterations=iterations, invalid_action_setting=invalid_action_setting)
        datapoints.append((true_utility, misspecified_reward, on_tomato))

    df = pd.DataFrame(datapoints, columns=["true_utility", "misspecified_reward", "steps_on_tomato"])
    df.to_csv(save_path, index=False)

    return

def sample_random_policy(steps: int = 1000, iterations: int = 0, invalid_action_setting: InvalidActionSetting = InvalidActionSetting.WAIT):
    grid = TomatoGrid(invalid_action_setting=invalid_action_setting)

    total_true_utility = 0
    total_misspecified_reward = 0
    total_on_tomato = 0

    if iterations > 0:
        sequence, _ = iterative_complexity_reduction(length=steps, iterations=iterations)
    else:
        sequence = [random.choice(list(Action)) for _ in range(steps)]

    for action in sequence:
        step_output = grid.update_grid(Action(action))
        total_true_utility += step_output.true_utility
        total_misspecified_reward += step_output.misspecified_reward
        total_on_tomato += step_output.on_tomato

    average_true_utility = total_true_utility / steps
    average_misspecified_reward = total_misspecified_reward / steps
    average_on_tomato = total_on_tomato / steps

    return average_true_utility, average_misspecified_reward, average_on_tomato


if __name__ == "__main__":
    main()
