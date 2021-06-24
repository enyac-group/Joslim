import torch
import torch.nn as nn

import numpy as np


class SlimBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SlimBN, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.active_ch = num_features

    def set_out_ch(self, ch):
        self.active_ch = ch

    def forward(self, x):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        weight = self.weight[:self.active_ch]
        bias = self.bias[:self.active_ch]
        running_mean = self.running_mean[:self.active_ch] if not self.training or self.track_running_stats else None
        running_var = self.running_var[:self.active_ch] if not self.training or self.track_running_stats else None
        return nn.functional.batch_norm(
                    x,
                    running_mean,
                    running_var,
                    weight, bias, bn_training, exponential_average_factor, self.eps)


class SlimConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SlimConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                         dilation, groups, bias, padding_mode)

        self.active_in_ch = in_channels
        self.active_out_ch = out_channels
        self.active_groups = groups
        self.out_h = 0
        self.out_w = 0

    def set_out_ch(self, ch):
        self.active_out_ch = ch

    def set_in_ch(self, ch):
        self.active_in_ch = ch
    
    def set_groups(self, ch):
        self.active_groups = ch

    def forward(self, x):
        weight = self.weight[:self.active_out_ch, :self.active_in_ch]
        out = nn.functional.conv2d(x, weight, None, self.stride,
                                   self.padding, self.dilation, self.active_groups)
        self.out_h = out.shape[2]
        self.out_w = out.shape[3]
        return out


class SlimDownsample(nn.Module):  
    def __init__(self, nIn, nOut, stride):
        super(SlimDownsample, self).__init__() 
        assert stride == 2    
        self.out_channels = nOut
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

    def forward(self, x):   
        x = self.avg(x)  
        if self.out_channels-x.size(1) > 0:
            pad = torch.zeros(x.size(0), self.out_channels-x.size(1), x.size(2), x.size(3)).to(x.device)
            return torch.cat((x, pad), 1) 
        else:
            return x

    def set_out_ch(self, ch):
        self.out_channels = ch
        return ch

    def set_in_ch(self, ch):
        pass


class SlimDownsampleConv(nn.Module):  
    def __init__(self, nIn, nOut, stride, norm_layer):
        super(SlimDownsampleConv, self).__init__() 
        self.conv = SlimConv2d(nIn, nOut, stride=stride, kernel_size=1)
        self.bn = norm_layer(nOut)

    def forward(self, x):   
        return self.bn(self.conv(x))

    def set_out_ch(self, ch):
        self.conv.set_out_ch(ch)
        self.bn.set_out_ch(ch)
        return ch

    def set_in_ch(self, ch):
        self.conv.set_in_ch(ch)
        return ch


class SlimBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SlimBasicBlock, self).__init__()
        self.conv = nn.Sequential(
            SlimConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            SlimBN(planes),
            nn.ReLU(inplace=True),
            SlimConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            SlimBN(planes),
        )
        self.downsample = downsample
        self.search_dim = 2

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = nn.functional.relu(out)
        return out

    def set_out_ch(self, ch):
        if isinstance(ch, int):
            ch = [ch for _ in range(self.search_dim)]
        assert len(ch) == self.search_dim
        if hasattr(self.downsample, 'set_out_ch'):
            self.downsample.set_out_ch(ch[1])
        self.conv[0].set_out_ch(ch[0])
        self.conv[1].set_out_ch(ch[0])
        self.conv[3].set_in_ch(ch[0])
        self.conv[3].set_out_ch(ch[1])
        self.conv[4].set_out_ch(ch[1])

    def set_in_ch(self, ch):
        self.conv[0].set_in_ch(ch)
        if hasattr(self.downsample, 'set_in_ch'):
            self.downsample.set_in_ch(ch)


class SlimBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SlimBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = SlimBN
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SlimConv2d(inplanes, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.conv2 = SlimConv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = SlimConv2d(width, planes * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.search_dim = 2

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def set_out_ch(self, ch):
        if isinstance(ch, int):
            ch = [ch for _ in range(self.search_dim)]
        assert len(ch) == self.search_dim
        if hasattr(self.downsample, 'set_out_ch'):
            self.downsample.set_out_ch(ch[-1])
        self.conv1.set_out_ch(ch[0])
        self.bn1.set_out_ch(ch[0])

        self.conv2.set_in_ch(ch[0])
        self.conv2.set_out_ch(ch[0])
        self.bn2.set_out_ch(ch[0])

        self.conv3.set_in_ch(ch[0])
        self.conv3.set_out_ch(ch[1])
        self.bn3.set_out_ch(ch[1])

    def set_in_ch(self, ch):
        self.conv1.set_in_ch(ch)
        if hasattr(self.downsample, 'set_in_ch'):
            self.downsample.set_in_ch(ch)


class SlimDWSConv(nn.Module):
    def __init__(self, inp, oup, stride, kernel=3, se=False, nl='RE'):
        super(SlimDWSConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            SlimConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            SlimBN(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            SlimConv2d(inp, oup, 1, 1, 0, bias=False),
            SlimBN(oup),
        )

        self.search_dim = 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def set_out_ch(self, ch):
        if isinstance(ch, int):
            ch = [ch for _ in range(self.search_dim)]
        assert len(ch) == self.search_dim
        self.conv[3].set_out_ch(ch[0])
        self.conv[4].set_out_ch(ch[0])

    def set_in_ch(self, ch):
        self.conv[0].set_in_ch(ch)
        self.conv[0].set_out_ch(ch)
        self.conv[0].set_groups(ch)
        self.conv[1].set_out_ch(ch)
        self.conv[3].set_in_ch(ch)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.functional.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu6(x + 3., inplace=self.inplace) / 6.


class SlimSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SlimSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            SlimConv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            SlimConv2d(channel // reduction, channel, kernel_size=1, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

    def set_out_ch(self, ch):
        self.fc[2].set_out_ch(ch)

    def set_in_ch(self, ch):
        self.fc[0].set_in_ch(ch)
    

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SlimInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, mid_c, kernel=3, se=False, nl='RE'):
        super(SlimInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert kernel in [3, 5, 7]
        padding = (kernel - 1) // 2
        self.use_res_connect = self.stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = nn.ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SlimSEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            SlimConv2d(inp, mid_c, 1, 1, 0, bias=False),
            SlimBN(mid_c),
            nlin_layer(inplace=True),
            # dw
            SlimConv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
            SlimBN(mid_c),
            SELayer(mid_c),
            nlin_layer(inplace=True),
            # pw-linear
            SlimConv2d(mid_c, oup, 1, 1, 0, bias=False),
            SlimBN(oup),
        )

        self.search_dim = 2

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def set_out_ch(self, ch):
        if isinstance(ch, int):
            ch = [ch for _ in range(self.search_dim)]
        assert len(ch) == self.search_dim
        # PW
        self.conv[0].set_out_ch(ch[0])
        self.conv[1].set_out_ch(ch[0])
        # DW
        self.conv[3].set_in_ch(ch[0])
        self.conv[3].set_out_ch(ch[0])
        self.conv[3].set_groups(ch[0])
        self.conv[4].set_out_ch(ch[0])
        # SE
        if hasattr(self.conv[5], 'set_in_ch'):
            self.conv[5].set_in_ch(ch[0])
            self.conv[5].set_out_ch(ch[0])
        # PW
        self.conv[7].set_in_ch(ch[0])
        self.conv[7].set_out_ch(ch[1])
        self.conv[8].set_out_ch(ch[1])

    def set_in_ch(self, ch):
        self.conv[0].set_in_ch(ch)


class DynModel(nn.Module):
    # This is the number of searchable dimensions
    search_dim = 1
    # This is the width multiplier lower bound for each layer
    lower_bnd = 0.2
    # This is the width multiplier upper bound for each layer
    # NOTE: This cannot be greater than 1!
    upper_bnd = 1

    def random_sample_wm(self, set_arch=True):
        wm = (np.random.rand(self.search_dim)) * (self.upper_bnd-self.lower_bnd) + self.lower_bnd
        wm = wm.tolist()
        real_ch = self.decode_wm(wm)
        if set_arch:
            self.set_real_ch(real_ch)
        return real_ch

    def decode_ch(self, ch):
        """ This function expands `ch` to a real network configuration

        ch - the searchable dimensions in [0, C_l] where C_l is the channel
        counts for the l-th layer

        Return a real network configuration by expanding `ch` to proper dimension and values.
        The return values are real channel counts as opposed to width multipliers.
        The results will be used by `self.set_real_ch`

        """
        pass

    def decode_wm(self, wm):
        """ This function expands `wm` to a real network configuration

        wm - the searchable dimensions in [0, 1] (the output of any optimizer on network architectures)

        Return a real network configuration by expanding `wm` to proper dimension and values.
        The return values are real channel counts as opposed to width multipliers.
        The results will be used by `self.set_real_ch`

        """
        pass

    def set_real_ch(self, real_ch):
        """ This functiion set the current model to have the channel counts specified by `real_ch`

        real_ch - the channel counts for each layer

        This function should implement the logic of setting each layer's channel counts appropriately.
        The key part is to ensure the input/output channel counts match across layers.
        
        """
        pass

    def get_flops_from_wm(self, wm):
        self.set_real_ch(self.decode_wm(wm))
        mac = 0
        for m in self.modules():
            if isinstance(m, SlimConv2d):
                mac += (m.kernel_size[0] * m.kernel_size[1] * m.active_in_ch * m.active_out_ch * m.out_w * m.out_h) / m.active_groups
        return mac

    def get_flops_from_ch(self, ch):
        self.set_real_ch(self.decode_ch(ch))
        mac = 0
        for m in self.modules():
            if isinstance(m, SlimConv2d):
                mac += (m.kernel_size[0] * m.kernel_size[1] * m.active_in_ch * m.active_out_ch * m.out_w * m.out_h) / m.active_groups
        return mac