import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict

def main():

    plot_q_matrix_data()


def plot_q_matrix_data() -> None:
    df_13 = pd.read_csv("Q Matrix Solving/Data/results_reward_13.csv")
    df_none = pd.read_csv("Q Matrix Solving/Data/results_reward_None.csv")

    name_dict = defaultdict(str)

    name_dict["boltzmann_sampling"] = "Boltzmann Sampling"
    name_dict["soft_q_learning"] = "Soft Q-Learning"
    name_dict[f"q_cap_t_inv_{10/13}"] = "Q-Capping"
    name_dict[f"q_cap_soft_t_inv_{10/13}"] = "Quantilization (ours)"

    df_13["Name"] = df_13["category"].map(name_dict)

    sns.lineplot(data=df_13[df_13["Name"] != ""], x="reward", y="utility", hue="Name")
    sns.scatterplot(data=df_13[df_13["Name"] != ""], x="reward", y="utility", hue="Name", legend=False)


    sns.lineplot(data=df_none, x="reward", y="utility", color="black")
    
    plt.show()



if __name__ == "__main__":
    main()