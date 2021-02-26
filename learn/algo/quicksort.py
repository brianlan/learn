from typing import List


def turn(l, r, active):
    if active == "l":
        return l, r - 1, "r"
    elif active == "r":
        return l + 1, r, "l"
    else:
        raise


def move(l, r, active):
    if active == "l":
        return l + 1, r
    elif active == "r":
        return l, r - 1
    else:
        raise


def qsort_(nums: List, lo, hi) -> None:
    if len(nums) <= 1 or hi - lo < 1:
        return

    l, r = lo, hi  # init moving pointer (active) and the reference pointer
    active = "l"
    while l != r:
        if nums[l] > nums[r]:
            nums[l], nums[r] = nums[r], nums[l]
            l, r, active = turn(l, r, active)
        else:
            l, r = move(l, r, active)
    if l - lo > 1:
        qsort_(nums, lo, l-1)
    if hi - l > 1:
        qsort_(nums, l+1, hi)
