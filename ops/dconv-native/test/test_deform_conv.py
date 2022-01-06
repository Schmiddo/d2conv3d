import unittest
import torch
from torch.nn.functional import conv1d, conv3d
from torch.nn.modules.utils import _single, _triple
from torch.autograd import gradcheck

from functools import reduce
from itertools import product
from math import floor

from dconv_native import deform_conv1d, deform_conv3d


def prod(t):
    return reduce(lambda a, b: a * b, t)


def linear_interpolate(data, x):
    x0 = int(floor(x))
    x1 = x0 + 1

    val = 0
    if 0 <= x0 < data.shape[0]:
        val += data[x0] * (x1 - x)
    if 0 <= x1 < data.shape[0]:
        val += data[x1] * (x - x0)

    return val


def trilinear_interpolate(data, z, y, x):
    z0 = int(floor(z))
    y0 = int(floor(y))
    x0 = int(floor(x))
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    z_ = (z0, z1)
    y_ = (y0, y1)
    x_ = (x0, x1)

    w = torch.zeros(2, 2, 2, device="cpu", dtype=torch.double)
    for i, j, k in product((0, 1), (0, 1), (0, 1)):
        w[i, j, k] = abs(z_[1-i] - z) * abs(y_[1-j] - y) * abs(x_[1-k] - x)

    def in_volume(p, d):
        return 0 <= p[0] < d[0] and 0 <= p[1] < d[1] and 0 <= p[2] < d[2]

    val = 0
    for i, j, k in product((0, 1), (0, 1), (0, 1)):
        if in_volume((z_[i], y_[j], x_[k]), data.shape[-3:]):
            val += w[i, j, k] * data[z_[i], y_[j], x_[k]]

    return val


def generate_data(
        N,
        batch_size=2,
        channels_in=3,
        channels_out=2,
        dimensions=5,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        n_offset_groups=1,
        n_weight_groups=1,
        bias=True,
        requires_grad=False,
        dtype=torch.float64,
        device="cuda"
):
    if N == 1:
        _t = _single
    elif N == 3:
        _t = _triple
    else:
        raise NotImplementedError(f"{N} dimensions not supported")
    dimensions = _t(dimensions)
    kernel_size = _t(kernel_size)
    stride = _t(stride)
    padding = _t(padding)
    dilation = _t(dilation)

    output_dimensions = tuple(
        (d + 2 * p - (a * (k - 1) + 1)) // s + 1
        for d, k, s, p, a
        in zip(dimensions, kernel_size, stride, padding, dilation)
    )

    common_args = {"requires_grad": requires_grad, "dtype": dtype, "device": device}
    num_kernel_points = prod(kernel_size)

    data = torch.randn(
        batch_size,
        channels_in,
        *dimensions,
        **common_args
    )
    offset = torch.randn(
        batch_size,
        n_offset_groups * N * num_kernel_points,
        *output_dimensions,
        **common_args
    )
    alpha = torch.randn(
        batch_size,
        n_offset_groups * num_kernel_points,
        *output_dimensions,
        **common_args
    )
    weight = torch.randn(
        channels_out,
        channels_in // n_weight_groups,
        *kernel_size,
        **common_args
    )
    if bias:
        bias = torch.randn(channels_out, **common_args)
    else:
        bias = None

    return data, offset, alpha, weight, bias


def expected_fn(N, x, offset, alpha, weight, bias, stride=1, padding=0, dilation=1):
    if N == 1:
        _t = _single
        interpolate_fn = linear_interpolate
    elif N == 3:
        _t = _triple
        interpolate_fn = trilinear_interpolate
    else:
        raise NotImplementedError(f"{N} dimensions not supported")
    stride = _t(stride)
    padding = _t(padding)
    dilation = _t(dilation)
    kernel_size = weight.shape[-N:]
    dimension = x.shape[-N:]

    n_batches, n_in_channels = x.shape[:2]
    n_out_channels = weight.shape[0]

    out_dims = tuple(
        (d + 2 * p - (a * (k - 1) + 1)) // s + 1
        for d, k, s, p, a
        in zip(dimension, kernel_size, stride, padding, dilation)
    )

    n_offset_grps = offset.shape[1] // (N * prod(kernel_size))

    n_weight_grps = n_in_channels // weight.shape[1]
    in_c_per_weight_grp = weight.shape[1]
    out_c_per_weight_grp = n_out_channels // n_weight_grps
    in_c_per_offset_grp = n_in_channels // n_offset_grps

    out = torch.zeros(n_batches, n_out_channels, *out_dims, device=x.device, dtype=x.dtype)
    for b in range(n_batches):
        for c_out in range(n_out_channels):
            for out_idx in product(*tuple(range(d) for d in out_dims)):
                for kernel_idx in product(*tuple(range(k) for k in kernel_size)):
                    for c in range(in_c_per_weight_grp):
                        weight_grp = c_out // out_c_per_weight_grp
                        c_in = weight_grp * in_c_per_weight_grp + c

                        offset_grp = c_in // in_c_per_offset_grp

                        alpha_idx = reduce(
                            lambda idx, n: idx * n[0] + n[1],
                            zip(kernel_size[1:], kernel_idx[1:]),
                            kernel_idx[0]
                        )
                        alpha_idx += offset_grp * prod(kernel_size)
                        offset_idx = N * alpha_idx

                        p = tuple(
                            s * l - p + a * dl + offset[(b, offset_idx + o) + out_idx]
                            for s, l, p, a, dl, o
                            in zip(stride, out_idx, padding, dilation, kernel_idx, range(N))
                        )

                        out[(b, c_out) + out_idx] += (
                                alpha[(b, alpha_idx) + out_idx]
                                * weight[(c_out, c) + kernel_idx]
                                * interpolate_fn(x[b, c_in], *p)
                        )

    if bias is not None:
        out += bias.view(1, n_out_channels, *N*(1,))
    return out


class DeformConvNdTesterMixin:
    dimensionality = 0
    function_under_test = None

    def generate_data(self, *args, **kwargs):
        return generate_data(self.dimensionality, *args, **kwargs)

    def expected_fn(self, *args, **kwargs):
        return expected_fn(self.dimensionality, *args, **kwargs)

    def _test_forward(self, input, offset, alpha, weight, bias=None, stride=1, padding=0,
                      dilation=1, n_weight_groups=1, n_offset_groups=1):
        output = self.function_under_test(input, offset, alpha, weight, bias, stride, padding, dilation, n_weight_groups, n_offset_groups)
        expected = expected_fn(self.dimensionality, input, offset, alpha, weight, bias, stride, padding, dilation)

        max_diff = (expected - output).abs().max()

        self.assertTrue(
            torch.allclose(expected, output),
            f"Forward not as expected. Max difference is {max_diff.item()}"
        )

    def _test_backward(self, x, offset, alpha, weight, bias=None, stride=1, padding=0, dilation=1, n_weight_groups=1, n_offset_groups=1):
        for t in (x, offset, alpha, weight, bias):
            if t is not None: t.requires_grad_(True)

        def func(x_, offset_, alpha_, weight_, bias_):
            return self.function_under_test(x_, offset_, alpha_, weight_, bias_, stride, padding, dilation, n_weight_groups, n_offset_groups)

        self.assertTrue(
            gradcheck(
                func,
                (x, offset, alpha, weight, bias),
                nondet_tol=1e-6
            )
        )


class DeformConv1dTester(DeformConvNdTesterMixin, unittest.TestCase):
    dimensionality = 1
    function_under_test = staticmethod(deform_conv1d)

    def test_no_deformation(self):
        data, offset, alpha, weight, bias = self.generate_data(
            dimensions=256,
            kernel_size=3,
            channels_in=24,
            channels_out=6,
            n_weight_groups=3
        )
        offset = torch.zeros_like(offset)
        alpha = torch.ones_like(alpha)

        out_conv = conv1d(data, weight, bias, groups=3)
        out_deform_conv = deform_conv1d(data, offset, alpha, weight, bias, n_weight_groups=3)

        max_diff = (out_conv - out_deform_conv).abs().max()

        self.assertTrue(
            torch.allclose(out_deform_conv, out_conv, atol=1e-7),
            f"Max difference is {max_diff.item()}"
        )

    def test_stride(self):
        stride = 3
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=2,
            stride=stride
        )

        self._test_forward(x, offset, alpha, weight, bias, stride)
        self._test_backward(x, offset, alpha, weight, bias, stride)

    def test_offset_groups(self):
        x, offset, alpha, weight, _ = self.generate_data(
            channels_in=12,
            channels_out=4,
            n_offset_groups=2
        )

        self._test_forward(x, offset, alpha, weight, n_offset_groups=2)
        self._test_backward(x, offset, alpha, weight, n_offset_groups=2)

    def test_weight_groups(self):
        x, offset, alpha, weight, _ = self.generate_data(
            channels_in=12,
            channels_out=6,
            n_weight_groups=3
        )

        self._test_forward(x, offset, alpha, weight, n_weight_groups=3)
        self._test_backward(x, offset, alpha, weight, n_weight_groups=3)

    def test_groups(self):
        x, offset, alpha, weight, _ = self.generate_data(
            channels_in=24,
            channels_out=12,
            n_weight_groups=6,
            n_offset_groups=2
        )

        self._test_forward(x, offset, alpha, weight, n_weight_groups=6, n_offset_groups=2)
        self._test_backward(x, offset, alpha, weight, n_weight_groups=6, n_offset_groups=2)

    def test_forward(self):
        stride = 2
        padding = 3
        dilation = 2
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=3,
            channels_in=3,
            channels_out=5,
            dimensions=34,
            kernel_size=5,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self._test_forward(x, offset, alpha, weight, bias, stride, padding, dilation)

    def test_backward(self):
        stride = 2
        padding = 3
        dilation = 2
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=4,
            channels_in=3,
            channels_out=5,
            dimensions=45,
            kernel_size=5,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self._test_backward(x, offset, alpha, weight, bias, stride, padding, dilation)

    def test_dtype_conversion(self):
        inputs = self.generate_data(bias=False, dtype=torch.half)
        result = self.function_under_test(inputs[0].float(), *inputs[1:])
        assert result.dtype == torch.float

    def test_no_bias(self):
        inputs = self.generate_data(bias=False)
        self._test_forward(*inputs)
        self._test_backward(*inputs)


class DeformConv3dTester(DeformConvNdTesterMixin, unittest.TestCase):
    dimensionality = 3
    function_under_test = staticmethod(deform_conv3d)

    def test_no_deformation(self):
        data, offset, alpha, weight, bias = self.generate_data(kernel_size=3)
        offset = torch.zeros_like(offset)
        alpha = torch.ones_like(alpha)

        #data = torch.ones_like(data)
        #weight = torch.ones_like(weight) * 3
        #bias = torch.zeros_like(bias)

        out_conv = conv3d(data, weight, bias)
        out_deform_conv = deform_conv3d(data, offset, alpha, weight, bias)

        #print(data)
        #print(weight)
        #print(bias)
        #print(out_conv)
        #print(out_deform_conv)
        #print(out_conv - out_deform_conv)

        max_diff = (out_conv - out_deform_conv).abs().max()

        self.assertTrue(
            torch.allclose(out_deform_conv, out_conv, atol=1e-7),
            f"Max difference is {max_diff.item()}"
        )

    def test_stride(self):
        stride = 3
        data = self.generate_data(
            stride=stride
        )

        self._test_forward(*data, stride=stride)
        self._test_backward(*data, stride=stride)

    def test_offset_groups(self):
        data = self.generate_data(
            channels_in=6,
            channels_out=4,
            n_offset_groups=2
        )

        self._test_forward(*data, n_offset_groups=2)
        self._test_backward(*data, n_offset_groups=2)

    def test_weight_groups(self):
        data = self.generate_data(
            channels_in=12,
            channels_out=6,
            n_weight_groups=3
        )

        self._test_forward(*data, n_weight_groups=3)
        self._test_backward(*data, n_weight_groups=3)

    def test_groups(self):
        data = self.generate_data(
            channels_in=24,
            channels_out=12,
            n_weight_groups=6,
            n_offset_groups=2
        )

        self._test_forward(*data, n_weight_groups=6, n_offset_groups=2)
        self._test_backward(*data, n_weight_groups=6, n_offset_groups=2)

    def test_channel_sep_groups(self):
        data = self.generate_data(
            channels_in=6,
            channels_out=6,
            n_weight_groups=6,
            n_offset_groups=3
        )

        self._test_forward(*data, n_weight_groups=6, n_offset_groups=3)
        self._test_backward(*data, n_weight_groups=6, n_offset_groups=3)

    def test_forward(self):
        stride = (1, 2, 1)
        padding = (0, 1, 2)
        dilation = (1, 1, 2)
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=2,
            channels_in=2,
            channels_out=3,
            dimensions=(5, 7, 7),
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self._test_forward(x, offset, alpha, weight, bias, stride, padding, dilation)

    def test_large_batch(self):
        """May take up to 20 minutes"""
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=21,
            channels_in=2,
            channels_out=2,
            dimensions=(3, 5, 5),
            kernel_size=(1, 3, 3)
        )

        self._test_forward(x, offset, alpha, weight, bias)
        self._test_backward(x, offset, alpha, weight, bias)

    def test_backward(self):
        stride = (1, 2, 1)
        padding = (1, 3, 5)
        dilation = (1, 1, 2)
        x, offset, alpha, weight, bias = self.generate_data(
            batch_size=2,
            channels_in=3,
            channels_out=5,
            dimensions=(2, 3, 5),
            kernel_size=(3, 5, 3),
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        self._test_backward(x, offset, alpha, weight, bias, stride, padding, dilation)

    def test_dtype_conversion(self):
        inputs = self.generate_data(bias=False, dtype=torch.half)
        result = self.function_under_test(inputs[0].float(), *inputs[1:])
        assert result.dtype == torch.float

    def test_no_bias(self):
        inputs = self.generate_data(bias=False)
        self._test_forward(*inputs)
        self._test_backward(*inputs)

    def test_no_alpha(self):
        input, offset, _, weight, bias = self.generate_data()
        #self._test_forward(input, offset, None, weight, bias)
        self._test_backward(input, offset, None, weight, bias)

    def test_known_offset(self):
        input, offsets, alpha, weight, bias = self.generate_data()
        offsets = torch.ones_like(offsets)
        alpha = torch.ones_like(alpha)

        shifted_input = torch.nn.functional.pad(
            input[:,:, 1:, 1:, 1:],
            (0, 1, 0, 1, 0, 1)
        )
        expected = conv3d(shifted_input, weight, bias)

        result = self.function_under_test(input, offsets, alpha, weight, bias)

        max_diff = (expected - result).abs().max()

        self.assertTrue(
            torch.allclose(expected, result, atol=1e-7),
            f"Max difference is {max_diff.item()}"
        )
        self.assertTrue(torch.allclose(expected, result))


if __name__ == '__main__':
    unittest.main()
