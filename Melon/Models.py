import torch
import torchvision
import torch.nn as nn

class PONO(nn.Module):
    # MOMENTS
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MomentShortcut(nn.Module):
    # MOMENTS
    def __init__(self, beta=None, gamma=None):
        super(MomentShortcut, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, num_channels=0, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg',pono=True,ms=True):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # MOMENTS
        self.pono = PONO(affine=False) if pono else None
        self.ms = MomentShortcut() if ms else None


    # def _forward_impl(self, x, input2=None):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     if input2 is not None:
    #       x2 = self.conv1(input2)
    #       x,_,_ = self.pono(x)
    #       x2, mean, std = self.pono(x2)
    #       x = self.ms(x,mean,std)
    #     else:
    #         x = self.bn1(x)  # TODO if bn is required or not
    #     x = self.relu(x)

    #     for layer in self.layers:
    #         x = layer(x)

    #     x = self.pool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x
    def forward(self, x, input2=None):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if input2 is not None:
          x2 = self.conv1(input2)
          x,_,_ = self.pono(x)
          x2, mean, std = self.pono(x2)
          x = self.ms(x,mean,std)
        # else:
        x = self.bn1(x)  # TODO if bn is required or not
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x