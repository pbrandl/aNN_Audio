class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Hardtanh()

    def forward(self, x):
        x = nn.functional.pad(x, (self.dilation, 0))
        f = self.conv_f(x)
        return torch.mul(self.conv_f(x), self.tanh(self.conv_g(x)))


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)
        return residual, skip



class WaveNet(nn.Module):
    def __init__(self, num_time_samples, num_channels=1, num_blocks=2, max_dilation=14,
                 num_hidden=32, kernel_size=2, device='cuda'):
        super(WaveNet, self).__init__()
        
        self.input_length = 0
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.max_dilation = max_dilation
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.device = device

        self.length_rf = (kernel_size - 1) * num_blocks * (1+ sum([2 ** k for k in range(max_dilation)]))
        self.previous_rf = None # Initial Receptive Field
        self.x_shape = None # Remember the input shape

        stacked_dilation = []

        first = True
        for b in range(num_blocks):
            for i in range(max_dilation):
                rate = 2 ** i
                if first:
                    hidden = GatedResidualBlock(num_channels, num_hidden, kernel_size, dilation=rate)
                    first = False
                else:
                    hidden = GatedResidualBlock(num_hidden, num_hidden, kernel_size, dilation=rate)
                    
                hidden.name = 'b{}-l{}'.format(b, i)
                stacked_dilation.append(hidden)
                #stacked_dilation.append(nn.Tanh())
                #batch_norms.append(nn.BatchNorm1d(num_hidden))

        self.stacked_dilation = nn.ModuleList(stacked_dilation)
        
        self.tanh = nn.Tanh()

        self.linear_mix = nn.Conv1d(
            in_channels=num_hidden,
            out_channels=1,
            kernel_size=1,
        )

        self.to(device)

    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_previous_rf(self):
        self.previous_rf = None

    def forward(self, x):
        self.x_shape = x.shape

        if self.previous_rf is None:
            self.previous_rf = torch.zeros((x.shape[0], x.shape[1], self.length_rf)).to(device)

        # Concat the last receptive field from x_(i-1) to x_i
        x_tended = torch.cat((self.previous_rf, x), dim=2)
        self.previous_rf = x[:, :, -self.length_rf:]
        
        skips = []
        for layer in self.stacked_dilation:
            x_tended, skip = layer(x_tended)
            skips.append(skip)
        
        x_tended = reduce(torch.add, skips)

        return self.linear_mix(x_tended)[:, :, self.length_rf:] + x

    def predict_sequence(self, x_seq):
        assert x_seq.dim() == 2, "Expected two-dimensional input shape (channels, lengths)."
        
        # Initialize 
        self.reset_previous_rf()
        x_length = self.x_shape[-1]
        x_seq_length = x_seq.shape[-1]
        channels = x_seq.shape[0]
        x_seq = x_seq.reshape(1, channels, x_seq_length)

        # Pad the input, s.t. it fits to the model's input expections
        pad_size = x_length - x_seq_length % x_length
        x_seq_padded = F.pad(x_seq, (pad_size, 0), mode='constant', value=0)
        x_seq_padded_length = x_seq_padded.shape[-1]
        y_seq_padded = torch.zeros_like(x_seq_padded)

        for i in range(0, x_seq_length, x_length):
            x_slice_c0 = x_seq_padded[:, 0, i:i+x_length].unsqueeze(0)
            y_seq_padded[:, 0, i:i+x_length] = model(x_slice_c0)

            if channels == 2:
                x_slice_c1 = x_seq_padded[:, 1, i:i+x_length].unsqueeze(0)
                y_seq_padded[:, 1, i:i+x_length] = model(x_slice_c1)
            #print(y_seq_padded.shape)

        y_seq = y_seq_padded[:, :, pad_size:]

        assert x_seq.shape == y_seq.shape, "Expected input and output to be equal in shape."
        return y_seq.reshape(channels, x_seq_length)


    
