from utils.q_matrix import QMatrix, create_reward_tensor
import pandas as pd
from tqdm import tqdm
from itertools import product

if __name__ == "__main__":
    reward_tensor = create_reward_tensor(13)

    q_matrix = QMatrix(
        misspecified_reward_value=13,
        kl_divergence_penalty=1,
        gamma = 0.9,
        t_inv = 5/13,
        q_cap=8.5,
    )

    kwargs_list = [{
        "misspecified_reward_value": 13,
        "kl_divergence_penalty": None,
        "gamma": 0.9,
        "t_inv": 0,
        "q_cap": None,
        "category": "totally_random",
    }]

    t_invs = [0.1/13, 0.2/13, 0.5/13, 1/13, 1.5/13, 2/13, 3/13]
    gammas = [1-1/100]

    for t_inv, gamma in product(t_invs, gammas):
        # Boltzmann sampling
        kwargs_list.append({
            "misspecified_reward_value": 13,
            "kl_divergence_penalty": None,
            "gamma": gamma,
            "t_inv": t_inv,
            "q_cap": None,
            "category": "boltzmann_sampling",
        })
        # Soft Q-learning
        kwargs_list.append({
            "misspecified_reward_value": 13,
            "kl_divergence_penalty": 1,
            "gamma": gamma,
            "t_inv": t_inv,
            "q_cap": None,
            "category": "soft_q_learning",
        })
        # Make misspecified reward higher
        kwargs_list.append({
            "misspecified_reward_value": 20,
            "kl_divergence_penalty": 1,
            "gamma": gamma,
            "t_inv": t_inv,
            "q_cap": None,
            "category": "soft_q_learning_high_reward",
        })

    q_caps = [6, 6.7, 7, 7.5, 8, 8.5, 9, 9.5, 10]

    for q_cap, gamma in product(q_caps, gammas):
        # Use just Q-cap
        kwargs_list.append({
            "misspecified_reward_value": 13,
            "kl_divergence_penalty": None,
            "gamma": gamma,
            "t_inv": 5/13,
            "q_cap": q_cap,
            "category": "q_cap",
        })
        # Use Q-cap and KL divergence penalty
        kwargs_list.append({
            "misspecified_reward_value": 13,
            "kl_divergence_penalty": 1,
            "gamma": gamma,
            "t_inv": 5/13,
            "q_cap": q_cap,
            "category": "q_cap_soft",
        })
        # Use Q-cap and make misspecified reward higher
        kwargs_list.append({
            "misspecified_reward_value": 20,
            "kl_divergence_penalty": 1,
            "gamma": gamma,
            "t_inv": 5/13,
            "q_cap": q_cap,
            "category": "q_cap_soft_high_reward",
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

    pd.DataFrame(dict_list).to_csv("Q Matrix Solving/Data/results.csv")

    print(status_list)