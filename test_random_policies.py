from utils import TomatoGrid
import pandas as pd
import os
import numpy as np
from itertools import product

PATH = "Random Policy Testing"

def main():

    thresholds = [0., 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    misspecified_reward_values = [13, 20]

    for threshold, misspecified_reward_value in product(thresholds, misspecified_reward_values):

        save_policy_datapoints(
            datapoints_count=100,
            save_path=f"Random Policy Testing/Data/datapoints_{misspecified_reward_value}_{threshold}.csv",
            steps=100,
            misspecified_reward_value=misspecified_reward_value,
            threshold=threshold)

def save_policy_datapoints(
        datapoints_count: int = 100,
        save_path: str = "datapoints.csv",
        misspecified_reward_value: float = 13,
        threshold: float = 0,
        steps: int = 100):

    if os.path.exists(save_path):
        print(f"Skipping {save_path} because it already exists")
        return
    else:
        print(f"Saving {save_path}")
    
    max_per_step_reward = min(misspecified_reward_value, 13)
    total_reward_threshold = threshold * steps
    
    datapoints: list[dict[str, float]] = []
    generator = np.random.default_rng(seed=12345)
    
    while(len(datapoints) < datapoints_count):
        grid = TomatoGrid(misspecified_reward_value=misspecified_reward_value, seed=42, invalid_action_setting="error")
        total_misspecified_reward = 0
        total_true_utility = 0

        for step_idx in range(steps):

            action = generator.choice(grid.get_valid_actions())
            output = grid.update_grid(action)
            total_misspecified_reward += output.misspecified_reward
            total_true_utility += output.true_utility

            # Check if it's possibile for us to reach the threshold
            if total_misspecified_reward + max_per_step_reward * (steps - step_idx) < total_reward_threshold:
                break

        if total_misspecified_reward >= total_reward_threshold:
            datapoints.append({
                "true_utility": total_true_utility / steps,
                "misspecified_reward": total_misspecified_reward / steps
            })


    df = pd.DataFrame(datapoints, columns=["true_utility", "misspecified_reward"])
    df.to_csv(save_path, index=False)

    return

if __name__ == "__main__":
    main()
