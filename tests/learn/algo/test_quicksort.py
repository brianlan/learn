import copy

import numpy as np

from learn.algo.quicksort import qsort_


def test_qsort():
    # nums = [8, 3, 1, 4, 6, 7, 5, 2, 5]
    # qsort_(nums)
    # assert nums == [1, 2, 3, 4, 5, 5, 6, 7, 8]
    for _ in range(1000):
        size = np.random.randint(20)
        nums = []
        if size > 0:
            nums = (np.random.randint(20, size=size) - 10).tolist()
        gt = copy.copy(nums)
        qsort_(nums, 0, len(nums) - 1)
        gt.sort()
        assert nums == gt
