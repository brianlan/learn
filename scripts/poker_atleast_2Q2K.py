from typing import List

from loguru import logger
from tqdm import tqdm
import numpy as np


def simulate(n_trials: int) -> float:
    success = 0
    deck = np.array([10] * 4 + [1] * 4 + [0] * 44)  # 10 denotes King and 1 denotes Queen
    for _ in tqdm(range(n_trials), total=n_trials):
        hands = np.random.choice(deck, size=5, replace=False)
        success += int(hands.sum() in [22, 23, 32])
    return success / n_trials


if __name__ == '__main__':
    prob = simulate(1000000)
    logger.info(f"prob by math: {6 * 6 * 48 / 52 / 51 / 50 / 49 / 48 * 5 * 4 * 3 * 2:.6f}")
    logger.info(f"prob by simulation: {prob:.6f}")
