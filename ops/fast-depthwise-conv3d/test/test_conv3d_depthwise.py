import unittest
from torch.testing._internal.common_utils import TestCase

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from functools import wraps

from fast_depthwise_conv3d import DepthwiseConv3d


def repeat_test_for_types(dtypes):
    def repeat_helper(f):
        @wraps(f)
        def call_helper(self, *args):
            for dtype in dtypes:
                with TestCase.subTest(self, dtype=dtype):
                    f(self, *args, dtype=dtype)

        return call_helper
    return repeat_helper


dtype2prec_DONTUSE = {torch.float: 1e-5,
                      torch.double: 1e-5,
                      torch.half: 1e-2,
                      torch.bfloat16: 1e-1}


class DepthwiseConv3dTester(TestCase):
    @repeat_test_for_types([torch.float, torch.double, torch.half])
    def test_Conv3d_depthwise_naive_groups_cuda(self, dtype=torch.float):
        for stride, dilation in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            for depth_multiplier, bias in [(1, True), (2, False)]:
                with self.subTest(stride=stride, dilation=dilation, depth_multiplier=depth_multiplier):
                    kernel_size = 3
                    padding = dilation * (kernel_size//2)
                    size_in = 6
                    size_out = (size_in + 2*(padding - dilation * (kernel_size//2))) // stride

                    m = DepthwiseConv3d(
                        2, 2 * depth_multiplier, kernel_size=kernel_size, bias=bias,
                        stride=stride, padding=padding, dilation=dilation
                    ).to("cuda", dtype)

                    i = torch.randn(2, 2, size_in, size_in, size_in, device="cuda", dtype=dtype).div_(2).requires_grad_()
                    output = m(i)
                    grad_output = torch.randn(2, 2 * depth_multiplier, size_out, size_out, size_out, device="cuda", dtype=dtype) / 2
                    output.backward(grad_output)

                    offset = 1 * depth_multiplier

                    m1 = nn.Conv3d(
                        1, 1 * depth_multiplier, kernel_size=kernel_size, bias=bias,
                        stride=stride, padding=padding, dilation=dilation
                    ).to("cuda", dtype)
                    m1.weight.data = m.weight.data[:offset].clone()
                    if bias:
                        m1.bias.data = m.bias.data[:offset].clone()
                    i1 = i.detach()[:, :1].clone().requires_grad_()
                    output1 = m1(i1)
                    output1.backward(grad_output[:, :offset].contiguous())

                    m2 = nn.Conv3d(
                        1, 1 * depth_multiplier, kernel_size=kernel_size, bias=bias,
                        stride=stride, padding=padding, dilation=dilation
                    ).to("cuda", dtype)
                    m2.weight.data.copy_(m.weight.data[offset:])
                    if bias:
                        m2.bias.data.copy_(m.bias.data[offset:])
                    i2 = i.detach()[:, 1:].clone().requires_grad_()
                    output2 = m2(i2)
                    output2.backward(grad_output[:, offset:].contiguous())

                    self.assertEqual(output, torch.cat([output1, output2], 1),
                                     atol=dtype2prec_DONTUSE[dtype], rtol=0)
                    self.assertEqual(i.grad.data,
                                     torch.cat([i1.grad.data, i2.grad.data], 1),
                                     atol=dtype2prec_DONTUSE[dtype], rtol=0)
                    if bias:
                        self.assertEqual(m.bias.grad.data,
                                     torch.cat([m1.bias.grad.data,
                                                m2.bias.grad.data], 0),
                                     atol=dtype2prec_DONTUSE[dtype], rtol=0)
                    self.assertEqual(m.weight.grad.data,
                                     torch.cat([m1.weight.grad.data,
                                                m2.weight.grad.data], 0),
                                     atol=dtype2prec_DONTUSE[dtype], rtol=0)

    def test_autograd_type_conversion(self):
        inp = torch.randn(2, 2, 5, 5, 5).to(dtype=torch.half, device="cuda")
        conv = DepthwiseConv3d(2, 2, kernel_size=3).cuda()
        with autocast(True):
            out = conv(inp)
            self.assertTrue(out.dtype == torch.half)



if __name__ == '__main__':
    unittest.main()
