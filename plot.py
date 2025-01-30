import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():

    plot_q_matrix_data()


def plot_q_matrix_data() -> None:
    df = pd.read_csv("Q Matrix Solving/Data/results.csv")

    categories = df["category"]

    reward_13 = df[
        (categories == "boltzmann_sampling") | \
        (categories == "soft_q_learning") | \
        (categories == "q_cap_soft")
    ]
    sns.lineplot(data=reward_13, x="reward", y="utility", hue="category")
    sns.scatterplot(data=df[df["category"] == "totally_random"], x="reward", y="utility", color="black")

    plt.show()



if __name__ == "__main__":
    main()