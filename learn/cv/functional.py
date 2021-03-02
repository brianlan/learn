from typing import Tuple

import numpy as np


def area(bboxes: np.array) -> np.ndarray:
    """Calculate the areas of given bounding boxes

    Parameters
    ----------
    bboxes : np.array
        of shape `(N, 4)`, coordinates in box: `[xmin, ymin, xmax, ymax]`
        pls. note, `(xmax - 1, ymax - 1)` is the bottom right pixel

    Returns
    -------
    np.ndarray
        of shape `(N, )`
    
    Examples
    --------
    .. code:: python
        >>> bboxes = np.array([[1, 2, 4, 6], [3, 0, 5, 4]])
        >>> areas = area(bboxes)
        >>> areas
        np.array([12, 8])
        
    """
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)
    w = (bboxes[:, 2] - bboxes[:, 0]).clip(min=0)
    h = (bboxes[:, 3] - bboxes[:, 1]).clip(min=0)
    return w * h


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate ious between 2 sets of bounding boxes.
    This function requires either of below conditions is True:
    - a.shape[0] == 1
    - b.shape[0] == 1
    - a.shape[0] == b.shape[0]

    Parameters
    ----------
    a : np.ndarray
        of shape `(N, 4)`, coordinates in box: `[xmin, ymin, xmax, ymax]`
        pls. note, `(xmax - 1, ymax - 1)` is the bottom right pixel
    b : np.ndarray
        of shape `(N, 4)`, coordinates in box: `[xmin, ymin, xmax, ymax]`
        pls. note, `(xmax - 1, ymax - 1)` is the bottom right pixel

    Returns
    -------
    np.ndarray
        ious of given sets of bounding boxes, of shape `(N, )`
    """
    if a.ndim == 1:
        a = np.expand_dims(a, axis=0)
    if b.ndim == 1:
        b = np.expand_dims(b, axis=0)
    assert a.shape[0] == 1 or a.shape[1] == 1 or a.shape[0] == b.shape[0]
    intersect = np.array(
        [
            np.maximum(a[:, 0], b[:, 0]),
            np.maximum(a[:, 1], b[:, 1]),
            np.minimum(a[:, 2], b[:, 2]),
            np.minimum(a[:, 3], b[:, 3]),
        ]
    ).transpose(1, 0)
    intersect_area = area(intersect)
    return intersect_area / (1e-6 + area(a) + area(b) - intersect_area)


def nms(bboxes: np.ndarray, iou_thresh: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression. Computational complexity: O(n^2). No sorting involved.

    Parameters
    ----------
    bboxes : np.ndarray
        bounding boxes of shape `(N, 5)`,
        each bbox consist of `[xmin, ymin, xmax, ymax, score]`
    iou_thresh : float, optional
        2 boxes with their iou higher than this value are considered similar in location, by default 0.7

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        indexes of kept bboxes and suppressed bboxes
    """
    if bboxes.ndim == 1:
        bboxes = np.expand_dims(bboxes, axis=0)
    kept_idx, suppressed_idx = [], []
    n = bboxes.shape[0]
    for i in range(n):
        ious = iou(bboxes[i], bboxes)
        exclude_self_mask = np.bool([True] * i + [False] + [True] * (n - i - 1))
        if ((ious > iou_thresh) & (bboxes[i, 4] < bboxes[:, 4]) & exclude_self_mask).sum() > 0:
            suppressed_idx.append(i)
        else:
            kept_idx.append(i)
    return np.array(kept_idx), np.array(suppressed_idx)


def conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dilation: int = 0,
) -> np.ndarray:
    """numpy version of conv2d, interface follows torch.nn.functional.conv2d

    Parameters
    ----------
    input : np.ndarray
        of shape `(N, C_in, H, W)`
    weight : np.ndarray
        of shape `(C_out, C_in, K, K)`, `K` is the kernel size.
    bias : np.ndarray, optional
        if given, the shape should be `(C_out,)`, by default None
    stride : int, optional
        by default 1
    padding : int, optional
        by default 0
    groups : int, optional
        by default 1
    dilation : int, optional
        by default 0

    Returns
    -------
    np.ndarray
        result tensor, of shape `(N, C_out, _, _)`,
        the spatial size depends on stride, padding.
    """
    if input.ndim == 3:
        input = np.expand_dims(input, axis=0)
    assert dilation == 0, "dilation > 0 not supported yet."
    assert input.ndim == weight.ndim
    assert weight.shape[1] * groups == input.shape[1]
    if bias is None:
        bias = np.zeros((weight.shape[0],))
    assert weight.shape[0] == bias.shape[0]
    assert weight.shape[2] == weight.shape[3], "non-equal kernel size not supported"
    C_out, _, K, _ = weight.shape
    padded_input = np.pad(
        input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), constant_values=0.0
    )
    N, C_in, H, W = padded_input.shape
    C_in_grp = C_in // groups  # C_in group size
    C_out_grp = C_out // groups  # C_out group size
    out = []
    for g in range(groups):
        input_g = padded_input[:, g * C_in_grp : (g + 1) * C_in_grp]
        weight_g = weight[g * C_out_grp : (g + 1) * C_out_grp, ...]
        bias_g = bias[g * C_out_grp : (g + 1) * C_out_grp]
        out_g = np.zeros((N, C_out_grp, (H - K + 1) // stride, (W - K + 1) // stride))
        for i in range((H - K + 1) // stride):
            for j in range((W - K + 1) // stride):
                si, sj = stride * i, stride * j
                input_block = input_g[:, None, :, si : si + K, sj : sj + K]
                out_g[:, :, i, j] = (input_block * weight_g).reshape(N, C_out_grp, -1).sum(
                    axis=2
                ) + bias_g[None, :]
        out.append(out_g)
    return np.concatenate(out, axis=1)
