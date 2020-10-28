import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super().forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


def _conv_stack(dilations, in_chann, out_chann, kernel_size) -> [CausalConv1d]:
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList([CausalConv1d(in_chann, out_chann, dilation=d, kernel_size=kernel_size) for d in dilations])


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * (np.sum([2 ** d for d in range(dilation_depth)]))
        print("Receptive field:", self.receptive_field)
        self.previous = None

        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        print("dilations", dilations)
        out_channels = int(num_channels * 2)

        self.hidden = _conv_stack(dilations, num_channels, out_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)

        self.input_layer = CausalConv1d(in_channels=1, out_channels=num_channels, kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )

        self.post_conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

        self.num_channels = num_channels

    def forward(self, x):
        if self.previous is None:
            self.previous = torch.zeros(x.shape[0], x.shape[1], self.receptive_field)

        x = torch.cat((self.previous, x), dim=2)
        self.previous = x[:, :, -self.receptive_field:]

        out = x
        skips = []
        out = self.input_layer(out)
        # print(x)
        # print(out)
        # print("shape input lay", out.shape)
        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # print("shape hidden lay", out_hidden.shape)

            # Gated Activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])
            # print("gated act", out.shape)

            skips.append(out)

            out = residual(out)
            # print("residual", out.shape, out.size(2))
            out = out + x
            # print("layer final", out.shape)

        # Modified "postprocess" step:
        out = torch.cat([s[:, :, :] for s in skips], dim=1)
        # print(out.shape)
        out = self.linear_mix(out)
        # print("post", out.shape)
        out = out[:, :, self.receptive_field:]

        return out
