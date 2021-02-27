from learn.cv.functional import conv2d

import numpy as np
import torch
import torch.nn.functional as F


def test_conv2d():
    input = torch.arange(400).reshape(8, 2, 5, 5) * 10
    weight = torch.arange(72).reshape(4, 2, 3, 3)
    out_npy = conv2d(input.numpy(), weight.numpy(), padding=1)
    out_pth = F.conv2d(input, weight, padding=1)
    np.testing.assert_almost_equal(out_npy, out_pth.numpy())

    input = torch.randn(8, 2, 5, 5)
    weight = torch.randn(4, 2, 3, 3)
    out_npy = conv2d(input.numpy(), weight.numpy(), padding=1)
    out_pth = F.conv2d(input, weight, padding=1)
    np.testing.assert_almost_equal(out_npy, out_pth.numpy(), decimal=5)


def test_conv2d_with_stride():
    input = torch.randn(8, 2, 6, 6)
    weight = torch.randn(4, 2, 3, 3)
    out_npy = conv2d(input.numpy(), weight.numpy(), padding=1, stride=2)
    out_pth = F.conv2d(input, weight, padding=1, stride=2)
    np.testing.assert_almost_equal(out_npy, out_pth.numpy(), decimal=5)


def test_conv2d_with_group():
    input = torch.randn(8, 2, 6, 6)
    weight = torch.randn(4, 1, 3, 3)
    out_npy = conv2d(input.numpy(), weight.numpy(), padding=1, groups=2)
    out_pth = F.conv2d(input, weight, padding=1, groups=2)
    np.testing.assert_almost_equal(out_npy, out_pth.numpy(), decimal=5)


def test_conv2d_with_stride_and_group():
    input = torch.randn(16, 8, 32, 32)
    weight = torch.randn(12, 4, 3, 3)
    out_npy = conv2d(input.numpy(), weight.numpy(), padding=1, groups=2, stride=2)
    out_pth = F.conv2d(input, weight, padding=1, groups=2, stride=2)
    np.testing.assert_almost_equal(out_npy, out_pth.numpy(), decimal=5)
