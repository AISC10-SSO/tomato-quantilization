import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from collections import defaultdict

def main():

    plot_demo_plots()

    plot_q_matrix_data()

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

def plot_q_matrix_data() -> None:
    df_13 = pd.read_csv("Q Matrix Solving/Data/results_reward_13.csv")
    df_none = pd.read_csv("Q Matrix Solving/Data/results_reward_None.csv")

    name_dict = defaultdict(str, {
        "boltzmann_sampling": "Boltzmann Sampling",
        "soft_q_learning": "Soft Q-Learning",
        f"q_cap_t_inv_{10/13}": "Q-Capping",
        f"q_cap_soft_t_inv_{10/13}": "Quantilization (ours)"
    })

    df_13["Name"] = df_13["category"].map(name_dict)

    sns.lineplot(data=df_13[df_13["Name"] != ""], x="reward", y="utility", hue="Name")
    sns.scatterplot(data=df_13[df_13["Name"] != ""], x="reward", y="utility", hue="Name", legend=False)

    sns.lineplot(data=df_none, x="reward", y="utility", color="black")
    
    plt.savefig("Q Matrix Solving/Plots/reward_utility_plot.png")
    plt.clf()

    def get_parameter_scale(row, dataframe):
        df_subset = dataframe[dataframe["Name"] == row["Name"]]
        if row["Name"] == "Quantilization (ours)" or row["Name"] == "Q-Capping":
            parameter = "q_cap"
        else:
            parameter = "t_inv"

        try:
            return (row[parameter] - min(df_subset[parameter])) / (max(df_subset[parameter]) - min(df_subset[parameter]))
        except:
            return 0

    df_13["parameter_scale"] = df_13.apply(
        lambda row: get_parameter_scale(row, df_13),
        axis=1
    )

    sns.lineplot(data=df_13[df_13["Name"] != ""], x="parameter_scale", y="utility", hue="Name")
    sns.scatterplot(data=df_13[df_13["Name"] != ""], x="parameter_scale", y="utility", hue="Name", legend=False)

    plt.savefig("Q Matrix Solving/Plots/reward_utility_plot_parameter_scale.png")
    plt.clf()

if __name__ == "__main__":
    main()