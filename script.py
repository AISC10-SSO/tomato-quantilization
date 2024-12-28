from utils import TomatoGrid, Action, calculate_complexity, iterative_complexity_reduction
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

def main():

    """
    for iterations in [0, 1, 2, 3, 4]:
        save_policy_datapoints(
            datapoints_count=10000,
            save_path=f"datapoints/datapoints_{iterations}_iterations.csv",
            iterations=iterations)
    """

    plot_policy_datapoints(save_path="plots", data_path="datapoints")

def plot_policy_datapoints(save_path: str, data_path: str):
    iterations_list = []
    true_utilities_list = []
    false_utilities_list = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            iterations = file.split("_")[1]
            title = f"{iterations} iterations"
            false_utility, true_utility = plot_single_policy_datapoints(
                read_path=f"{data_path}/{file}",
                save_path=f"{save_path}/{file}",
                title=title,
                penalty=0)
            
            false_utilities_list.append(false_utility)
            true_utilities_list.append(true_utility)
            iterations_list.append(int(iterations))

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            datapoints = pd.read_csv(f"{data_path}/{file}")
            iterations = file.split("_")[1]
            plt.scatter(datapoints["false_utility"], datapoints["true_utility"], alpha=0.1, label=f"{iterations} iterations")
    plt.xlabel("False Utility")
    plt.ylabel("True Utility")
    plt.title(f"All Datapoints")
    plt.legend()
    plt.savefig(f"{save_path}/all_datapoints.png")
    plt.clf()

    # Sort datapoints by iterations
    sorted_indices = sorted(range(len(iterations_list)), key=lambda k: iterations_list[k])
    iterations_list = [iterations_list[i] for i in sorted_indices]
    false_utilities_list = [false_utilities_list[i] for i in sorted_indices] 
    true_utilities_list = [true_utilities_list[i] for i in sorted_indices]

    plt.plot(iterations_list, false_utilities_list, label="False Utility")
    plt.plot(iterations_list, true_utilities_list, label="True Utility")
    plt.xlabel("Iterations")
    plt.ylabel("Utility")
    plt.title("Policy Datapoints")
    plt.legend()
    plt.savefig(f"{save_path}/policy_datapoints.png")
    plt.clf()

def plot_single_policy_datapoints(read_path: str, save_path: str, title: str, penalty: float = 0):
    df = pd.read_csv(read_path)
    plt.scatter(df["false_utility"], df["true_utility"], alpha=0.1)
    plt.xlabel("False Utility")
    plt.ylabel("True Utility")
    plt.title(title)
    plt.savefig(f"{save_path}.png")

    plt.clf()

    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    # Complexity is compression ratio, so penalty should be - 1/complexity
    df["complexity_penalty"] = -1 / df["complexity"] * penalty
    df["penalized_false_utility"] = df["false_utility"] + df["complexity_penalty"]

    quantile_true_utilities = []
    quantile_false_utilities = []

    for q in quantiles:
        quantile_true_utilities.append(df["true_utility"].quantile(q))
        quantile_false_utilities.append(df["penalized_false_utility"].quantile(q))

    plt.plot(quantiles, quantile_true_utilities, label="True Utility")
    plt.plot(quantiles, quantile_false_utilities, label="False Utility")
    plt.xlabel("Quantile")
    plt.ylabel("Utility")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{save_path}_{title}_quantiles.png")
    plt.clf()

    return quantile_true_utilities[-1], quantile_false_utilities[-1]

def save_policy_datapoints(datapoints_count: int = 10000, save_path: str = "datapoints.csv", iterations: int = 0):

    datapoints = []
    for _ in tqdm(range(datapoints_count)):
        true_utility, false_utility, complexity = sample_random_policy(iterations=iterations)
        datapoints.append((true_utility, false_utility, complexity))

    df = pd.DataFrame(datapoints, columns=["true_utility", "false_utility", "complexity"])
    df.to_csv(save_path, index=False)

def sample_random_policy(steps: int = 1000, iterations: int = 0):
    grid = TomatoGrid()

    total_true_utility = 0
    total_false_utility = 0

    if iterations > 0:
        sequence, complexity = iterative_complexity_reduction(length=steps, iterations=iterations)
        for action in sequence:
            grid.update_grid(Action(action))
            total_true_utility += grid.get_current_utility()[0]
            total_false_utility += grid.get_current_utility()[1]
    else:
        complexity = 1
        for _ in range(steps):
            action = random.choice(list(Action))
            grid.update_grid(action)
            total_true_utility += grid.get_current_utility()[0]
            total_false_utility += grid.get_current_utility()[1]

    average_true_utility = total_true_utility / steps
    average_false_utility = total_false_utility / steps

    return average_true_utility, average_false_utility, complexity


if __name__ == "__main__":
    main()
