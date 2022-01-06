import unittest
import torch
import torch.nn as nn

from dconv_native import DConv3d, MDConv3d, SizeConditionedDConv3d


class DeformConvTester(unittest.TestCase):
    def test_no_deform_dconv(self):
        data = torch.randn(2, 3, 7, 7, 7).cuda()

        conv = nn.Conv3d(3, 4, 3).cuda()
        dconv = DConv3d(3, 4, 3).cuda()

        dconv.deform_conv.weight.data = conv.weight.data
        dconv.deform_conv.bias.data = conv.bias.data

        out_conv = conv(data)
        out_deform_conv = dconv(data)

        max_diff = (out_conv - out_deform_conv).abs().max()

        self.assertTrue(
            torch.allclose(out_conv, out_deform_conv, atol=1e-6),
            f"Max difference is {max_diff.item()}"
        )

    def test_no_deform_mdconv(self):
        data = torch.randn(2, 3, 7, 7, 7).cuda()

        conv = nn.Conv3d(3, 4, 3).cuda()
        mdconv = MDConv3d(3, 4, 3).cuda()

        mdconv.deform_conv.weight.data = conv.weight.data
        mdconv.deform_conv.bias.data = conv.bias.data

        # Modulate with sigmoid(0) = 0.5
        out_conv = conv(data * 0.5)
        out_deform_conv = mdconv(data)

        max_diff = (out_conv - out_deform_conv).abs().max()

        self.assertTrue(
            torch.allclose(out_conv, out_deform_conv, atol=1e-6),
            f"Max difference is {max_diff.item()}"
        )

    def test_size_conditioning(self):
        data = torch.randn(2, 3, 7, 7, 7).cuda()

        conv = nn.Conv3d(3, 4, 3, padding=(1, 2, 2), dilation=(1, 2, 2)).cuda()
        dconv = SizeConditionedDConv3d(3, 4, 3, padding=1).cuda()

        dconv.deform_params.weight.data[0] = 0
        dconv.deform_params.bias.data[0] = 1
        dconv.deform_conv.weight.data = conv.weight.data
        dconv.deform_conv.bias.data = conv.bias.data

        out_conv = conv(data)
        out_deform_conv = dconv(data)

        max_diff = (out_conv - out_deform_conv).abs().max()

        self.assertTrue(
            torch.allclose(out_conv, out_deform_conv, atol=1e-6),
            f"Max difference is {max_diff.item()}"
        )

    def test_backward(self):
        data = torch.randn(2, 3, 7, 7, 7).cuda()
        dconv = DConv3d(3, 4, 3, padding=1, bias=False).cuda()
        loss = dconv(data).mean()
        loss.backward()
