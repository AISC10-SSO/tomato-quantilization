import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from collections import defaultdict

def main():

    # plot_demo_plots()

    for misspecified_reward_value in [13, 20]:
        # plot_q_matrix_data(misspecified_reward_value)
        pass

    # plot_q_matrix_comparison(misspecified_reward_values=[13, 20])

    for misspecified_reward_value in [13, 20]:
        plot_thresholded_trajectories(misspecified_reward_value)

def plot_thresholded_trajectories(misspecified_reward_value: int = 13) -> None:

    q_matrix_df = pd.read_csv(f"Q Matrix Solving/Data/results_reward_{misspecified_reward_value}.csv")
    q_matrix_df["misspecified_reward"] = q_matrix_df["reward"] / 100
    q_matrix_df["true_utility"] = q_matrix_df["utility"] / 100
    q_matrix_df["threshold"] = q_matrix_df["q_cap"]
    q_matrix_df = q_matrix_df[q_matrix_df["category"] == f"q_cap_soft_t_inv_{10/13}"][["misspecified_reward", "true_utility", "threshold"]]
    q_matrix_df["Name"] = "Q-Matrix Solver"

    df_list = []
    for threshold in [0., 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]:
        df = pd.read_csv(f"Random Policy Testing/Data/datapoints_{misspecified_reward_value}_{threshold}.csv")
        df["threshold"] = threshold
        df_list.append(df.mean(axis=0))

    monte_carlo_df = pd.DataFrame(df_list)
    monte_carlo_df["Name"] = "Monte Carlo"

    final_df = pd.concat([monte_carlo_df, q_matrix_df])


    # Plot reward against utility
    sns.lineplot(data=final_df, x="threshold", y="true_utility", style="Name")
    sns.lineplot(data=final_df, x="threshold", y="misspecified_reward", style="Name")

    plt.savefig(f"Random Policy Testing/Plots/thresholded_trajectories_{misspecified_reward_value}.png", bbox_inches="tight", dpi=600)
    plt.clf()


def plot_demo_plots() -> None:
    q = np.linspace(0, 20, 100)

    sns.set_palette("colorblind")
    df_list = []
    for t, q_cap in ((2, 5), (3, 5),(4, 5), (4, 8)):
        y = -np.log(1 + np.exp((q_cap-q) / t))
        df_list.append(pd.DataFrame({"q": q, "y": y, "T": t, r"$Q_{cap}$": q_cap}))

    sns.lineplot(data=pd.concat(df_list), x="q", y="y", hue="T", style=r"$Q_{cap}$")

    plt.xlabel(r"$Q_r$")
    plt.ylabel("Logit")
    plt.savefig("Q Matrix Solving/Plots/demo_plot_1.png")
    plt.clf()

def plot_q_matrix_comparison(misspecified_reward_values: list[int] = [13, 20]) -> None:

    df_dict = {
        misspecified_reward_value: pd.read_csv(f"Q Matrix Solving/Data/results_reward_{misspecified_reward_value}.csv")
        for misspecified_reward_value in misspecified_reward_values
    }

    df_none = pd.read_csv(f"Q Matrix Solving/Data/results_reward_None.csv")

    name_dict = defaultdict(str, {
        "boltzmann_sampling": "Boltzmann Sampling",
        "soft_q_learning": "Soft Q-Learning",
        f"q_cap_t_inv_{10/13}": "Q-Capping",
        f"q_cap_soft_t_inv_{10/13}": "Quantilization (ours)"
    })

    for df in df_dict.values():
        df["Name"] = df["category"].map(name_dict)

    max_values_dict = {
        misspecified_reward_value: df[df["Name"] != ""].groupby("Name").max()
        for misspecified_reward_value, df in df_dict.items()
    }


    min_utility = df_none["utility"].min()
    max_utility = df_none["utility"].max()

    normalize = lambda x: (x - min_utility) / (max_utility - min_utility)

    barchart_df = pd.DataFrame({
        "Method": list(max_values_dict[misspecified_reward_values[0]].index),
        **{
            str(misspecified_reward_value): list(normalize(max_values_dict[misspecified_reward_value]["utility"]))
            for misspecified_reward_value in misspecified_reward_values
        }
    })


    barchart_df.set_index(barchart_df["Method"], inplace=True)
    barchart_df = barchart_df.loc[["Boltzmann Sampling", "Soft Q-Learning", "Q-Capping", "Quantilization (ours)"]]

    data_melted = pd.melt(
        barchart_df, 
        id_vars=['Method'],
        value_vars=map(str, misspecified_reward_values),
        var_name='group',
        value_name='value')

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x='group', 
        y='value',
        hue='Method',
        data=data_melted)

    plt.xlabel("Value of the Reward Hack")
    plt.ylabel("Best Possible Performance (normalized)")
    plt.xticks([0, 1], ["13 (standard)", "20 (more difficult)"])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("Q Matrix Solving/Plots/reward_utility_plot_comparison.png", bbox_inches="tight", dpi=600)
    plt.clf()


def plot_q_matrix_data(misspecified_reward_value: int = 13) -> None:
    df = pd.read_csv(f"Q Matrix Solving/Data/results_reward_{misspecified_reward_value}.csv")
    df_none = pd.read_csv(f"Q Matrix Solving/Data/results_reward_None.csv")

    name_dict = defaultdict(str, {
        "totally_random": "Totally Random",
        "boltzmann_sampling": "Boltzmann Sampling",
        "soft_q_learning": "Soft Q-Learning",
        f"q_cap_t_inv_{10/13}": "Q-Capping",
        f"q_cap_soft_t_inv_{10/13}": "Quantilization (ours)"
    })

    df["Name"] = df["category"].map(name_dict)

    sns.lineplot(
        data=df[(df["Name"] != "") & (df["Name"] != "Totally Random")],
        x="reward",
        y="utility",
        hue="Name",
        markers=True)

    sns.lineplot(data=df_none, x="reward", y="utility", color="black", label="No Reward Misspecification")
    sns.scatterplot(data=df_none[-1:], x="reward", y="utility", color="black", label="Maximum Possible Performance")
    sns.scatterplot(data=df[df["Name"] == "Totally Random"], x="reward", y="utility", color="grey", label="Random Policy")

    if misspecified_reward_value == 20:
        plt.xlim(None, 1350)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(f"Q Matrix Solving/Plots/reward_utility_plot_{misspecified_reward_value}.png", bbox_inches="tight", dpi=600)
    plt.clf()

if __name__ == "__main__":
    main()