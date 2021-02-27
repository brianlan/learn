from learn.cv.functional import nms

import numpy as np


def test_nms():
    bboxes = np.array([
        [ 2,  1,  5,  5, 1.0],  # 0
        [ 2,  3, 10, 14, 0.9],  # 1
        [ 2,  5, 12, 13, 0.8],  # 2
        [ 4,  3, 11, 16, 0.7],  # 3
        [ 6,  7, 16, 18, 0.6],  # 4
        [10, 15, 29, 22, 0.5],  # 5
        [12, 14, 28, 20, 0.4],  # 6
        [11, 16, 31, 23, 0.3],  # 7
        [19, 11, 21, 26, 0.2],  # 8
        [18,  1, 22,  5, 0.1],  # 9
    ])
    kept_idx, suppressed_idx = nms(bboxes, iou_thresh=0.5)
    assert set(kept_idx.tolist()) == {0, 1, 4, 5, 8, 9}
    assert set(suppressed_idx.tolist()) == {2, 3, 6, 7}

    kept_idx, suppressed_idx = nms(bboxes, iou_thresh=0.6)
    assert set(kept_idx.tolist()) == {0, 1, 3, 4, 5, 6, 8, 9}
    assert set(suppressed_idx.tolist()) == {2, 7}

    kept_idx, suppressed_idx = nms(bboxes, iou_thresh=0.3)
    assert set(kept_idx.tolist()) == {0, 1, 4, 5, 8, 9}
    assert set(suppressed_idx.tolist()) == {2, 3, 6, 7}
