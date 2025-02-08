from utils.q_matrix import QMatrix
import pandas as pd
from tqdm import tqdm
from itertools import product
import os

def main():

    q_matrix = QMatrix(
        misspecified_reward_value=13,
        kl_divergence_penalty=None,
        gamma=0.99,
        t_inv=0,
        q_cap=None,
    )

    print(q_matrix.get_reward_and_utility())


    exit()

    for reward_value in [None, 13, 20]:
        status_list = run_experiment(reward_value=reward_value)
        if status_list is None:
            continue
        if "error" in status_list:
            print(f"Error for {reward_value} on run {status_list.index('error')}")

def run_experiment(*, reward_value: int|None = 13) -> list[str]|None:
    """
    Run a set of experiments for a given reward value.

    Args:
        reward_value: The reward value to use for the experiment.
            if reward_value is None, we don't worry about Q-capping or soft Q-learning,
            but do use an extended range of t_inv values for boltzmann sampling

    Returns:
        A list of statuses for the experiment, or None if the file already exists.
    """

    if os.path.exists(f"Q Matrix Solving/Data/results_reward_{reward_value}.csv"):
        print(f"Skipping {reward_value} because the file already exists")
        return None
    else:
        print(f"Running experiment for {reward_value}")

    kwargs_list = [{
        "misspecified_reward_value": reward_value,
        "kl_divergence_penalty": None,
        "gamma": 0.99,
        "t_inv": 0,
        "q_cap": None,
        "category": "totally_random",
    }]

    t_invs = [0.05/13, 0.1/13, 0.2/13, 0.3/13, 0.5/13, 0.6/13, 0.7/13, 0.8/13, 0.9/13, 1/13, 1.5/13, 2/13, 3/13]

    if reward_value is None: # Want more t_invs for the None case
        t_invs += [5/13, 10/13, 20/13, 50/13]

    kl_divergence_penalties = [None, 1] if reward_value is not None else [None]

    for t_inv, kl_divergence_penalty in product(t_invs, kl_divergence_penalties):
        # Boltzmann sampling
        kwargs_list.append({
            "misspecified_reward_value": reward_value,
            "kl_divergence_penalty": kl_divergence_penalty,
            "gamma": 0.99,
            "t_inv": t_inv,
            "q_cap": None,
            "category": "boltzmann_sampling" if kl_divergence_penalty is None else "soft_q_learning",
        })

    q_caps = [6, 6.7, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    # Only bother with Q-capping if the reward is not None

    for q_cap in (q_caps if reward_value is not None else []):
        for t_inv in [1/13, 2/13, 5/13, 10/13, 20/13] if reward_value == 13 else  [5/13, 10/13]:
            kwargs_list.append({
                "misspecified_reward_value": reward_value,
                "kl_divergence_penalty": None,
                "gamma": 0.99,
                "t_inv": t_inv,
                "q_cap": q_cap,
                "category": f"q_cap_t_inv_{t_inv}",
            })
            # Use Q-cap and KL divergence penalty
            kwargs_list.append({
                "misspecified_reward_value": reward_value,
                "kl_divergence_penalty": 1,
                "gamma": 0.99,
                "t_inv": t_inv,
                "q_cap": q_cap,
                "category": f"q_cap_soft_t_inv_{t_inv}",
            })

    dict_list = []
    status_list = []
    for kwargs in tqdm(kwargs_list):
        try:
            q_matrix = QMatrix(**{k: v for k, v in kwargs.items() if k != "category"})
            q_matrix.train()
            dict_list.append({**kwargs, **q_matrix.get_reward_and_utility(), "error": None})
            status_list.append("success")
        except Exception as e:
            dict_list.append({**kwargs, "reward": None, "utility": None, "error": str(e)})
            status_list.append("error")

    pd.DataFrame(dict_list).to_csv(f"Q Matrix Solving/Data/results_reward_{reward_value}.csv")

    return status_list

if __name__ == "__main__":
    main()