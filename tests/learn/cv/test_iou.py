from learn.cv.functional import iou, area

import pytest
import numpy as np


def test_area():
    x = np.array([1, 2, 4, 6])
    y = np.array([3, 0, 5, 4])
    assert pytest.approx(area(x)[0]) == 3 * 4
    assert pytest.approx(area(y)[0]) == 2 * 4
    assert pytest.approx(area(np.array([18,  7, 16,  5]))[0]) == 0
    assert pytest.approx(area(np.array([16,  7, 18,  5]))[0]) == 0
    assert pytest.approx(area(np.array([18,  5, 16,  7]))[0]) == 0
    

def test_iou_single():
    x = np.array([1, 2, 5, 7])
    y = np.array([3, 0, 6, 5])
    z = np.array([1, 8, 3, 9])
    a = np.array([2, 5, 3, 6])
    b = np.array([0, 4, 2, 6])
    c = np.array([1, 8, 3, 7])
    d = np.array([-1, -1, 3, 3])
    assert pytest.approx(iou(x, y)[0]) == 6 / 29
    assert pytest.approx(iou(y, x)[0]) == 6 / 29
    assert pytest.approx(iou(z, x)[0]) == 0
    assert pytest.approx(iou(a, x)[0]) == 1 / 20
    assert pytest.approx(iou(x, b)[0]) == 2 / 22
    assert pytest.approx(iou(c, x)[0]) == 0
    assert pytest.approx(iou(d, x)[0]) == 2 / 34
    assert pytest.approx(iou(z, z)[0]) == 1
    assert pytest.approx(iou(np.array([18,  1, 22,  5]), np.array([ 6,  7, 16, 18]))[0]) == 0


def test_iou_multiple():
    a = np.array([
        [1, 2, 5, 7],
        [3, 0, 6, 5],
        [1, 8, 3, 9],
        [2, 5, 3, 6],
        [1, 2, 5, 7],
        [1, 8, 3, 7],
        [-1, -1, 3, 3],
    ])
    b = np.array([
        [3, 0, 6, 5],
        [1, 2, 5, 7],
        [1, 2, 5, 7],
        [1, 2, 5, 7],
        [0, 4, 2, 6],
        [1, 2, 5, 7],
        [1, 2, 5, 7],
    ])
    np.testing.assert_almost_equal(iou(a, b), np.array([6 / 29, 6 / 29, 0, 1/ 20, 2/22,0,2/34]))