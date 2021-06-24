import torch
import torch.nn as nn
import torch.nn.functional as F

from .dyn_model import SlimBN, SlimConv2d, SlimDWSConv, SlimInvertedResidual, DynModel, Hswish


__all__ = ['slim_mobilenetv3_small', 'slim_mobilenetv3_large', 'slim_mobilenetv3_super']

class SlimConvBN(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, nlin_layer=nn.ReLU6):
        super(SlimConvBN, self).__init__()
        self.conv = SlimConv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.norm = SlimBN(oup)
        self.act = nlin_layer(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def set_out_ch(self, ch):
        self.conv.set_out_ch(ch)
        self.norm.set_out_ch(ch)

    def set_in_ch(self, ch):
        self.conv.set_in_ch(ch)

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class SlimStage(nn.Module):
    def __init__(self, input_channel, local_setting):
        super(SlimStage, self).__init__()
        k, exp, c, se, nl, s = local_setting[0]
        self.head = SlimInvertedResidual(input_channel, c, s, exp, k, se, nl)
        self.tail = []
        input_channel = c
        for i in range(1, len(local_setting)):
            k, exp, c, se, nl, s = local_setting[i]
            self.tail.append(SlimInvertedResidual(input_channel, c, s, exp, k, se, nl))
            input_channel = c
        self.tail = nn.Sequential(*self.tail)

    def forward(self, x):
        return self.tail(self.head(x))

    def set_in_ch(self, planes):
        self.head.set_in_ch(planes)

    def set_out_ch(self, ch):
        nconv = 2
        assert len(ch) == (len(self.tail)+1)*nconv

        self.head.set_out_ch(ch[:nconv])
        for idx, m in enumerate(self.tail):
            m.set_in_ch(ch[(idx+1)*nconv-1])
            m.set_out_ch(ch[(idx+1)*nconv:(idx+2)*nconv])
        return ch


class SlimMobileNetV3(DynModel):
    def __init__(self, n_class=1000, input_size=224, dropout=0, mode='small', width_mult=1.0):
        super(SlimMobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            self.channel_space = [16,
                                  [16],
                                  [24, 64, 72],
                                  [40, 72, 120, 120], 
                                  [80, 240, 200, 184, 184],
                                  [112, 480, 672],
                                  [160, 672, 960, 960]]
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            self.channel_space = [16,
                                  [16],
                                  [24, 72, 88],
                                  [40, 96, 240, 240], 
                                  [48, 120, 144],
                                  [96, 288, 576, 576]]
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        elif mode == 'super':
            self.channel_space = [16,
                                  [16],
                                  [24, 144, 144, 144, 144],
                                  [40, 240, 240, 240, 240], 
                                  [80, 480, 480, 480, 480],
                                  [112, 672, 672, 672, 672],
                                  [160, 960, 960, 960, 960]]
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [7, 144,  24,  False, 'RE', 2],
                [7, 144,  24,  False, 'RE', 1],
                [7, 144,  24,  False, 'RE', 1],
                [7, 144,  24,  False, 'RE', 1],
                [7, 240,  40,  True,  'RE', 2],
                [7, 240, 40,  True,  'RE', 1],
                [7, 240, 40,  True,  'RE', 1],
                [7, 240, 40,  True,  'RE', 1],
                [7, 480, 80,  False, 'HS', 2],
                [7, 480, 80,  False, 'HS', 1],
                [7, 480, 80,  False, 'HS', 1],
                [7, 480, 80,  False, 'HS', 1],
                [7, 672, 112, True,  'HS', 1],
                [7, 672, 112, True,  'HS', 1],
                [7, 672, 112, True,  'HS', 1],
                [7, 672, 112, True,  'HS', 1],
                [7, 960, 160, True,  'HS', 2],
                [7, 960, 160, True,  'HS', 1],
                [7, 960, 160, True,  'HS', 1],
                [7, 960, 160, True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        self.search_dim = 0
        for stage in self.channel_space:
            if isinstance(stage, int):
                self.search_dim += 1
            else:
                for c in stage:
                    self.search_dim += 1

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [SlimConvBN(3, input_channel, 3, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        pre_c = input_channel
        stage = []
        local_setting = []
        for k, exp, c, se, nl, s in mobile_setting:
            if pre_c != c:
                stage.append(local_setting)
                local_setting = []
            local_setting.append([k, make_divisible(exp * width_mult), make_divisible(c * width_mult), se, nl, s])
            pre_c = make_divisible(c * width_mult)
        if len(local_setting):
            stage.append(local_setting)

        for idx, local_setting in enumerate(stage):
            if idx == 0:
                k, exp, c, se, nl, s = local_setting[0]
                self.features.append(SlimDWSConv(input_channel, c, s, k, se, nl))
            else:
                self.features.append(SlimStage(input_channel, local_setting))
            input_channel = local_setting[-1][2]

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(SlimConvBN(input_channel, last_conv, 1, 1, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(SlimConvBN(input_channel, last_conv, 1, 1, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'super':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(SlimConvBN(input_channel, last_conv, 1, 1, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def decode_ch(self, ch):
        prev_ch = 0
        real_ch = []
        for idx, stage_ch in enumerate(ch):
            expanded_stage_ch = stage_ch
            if idx > 1:
                stage_ch[0] = max(stage_ch[0], prev_ch)
                expanded_stage_ch = []
                for ci in range(len(stage_ch)-1):
                    expanded_stage_ch.append(stage_ch[ci+1])
                    expanded_stage_ch.append(stage_ch[0])
                prev_ch = stage_ch[0]
            real_ch.append(expanded_stage_ch)
        return real_ch

    def decode_wm(self, wm):
        new_wm = []
        i = 0
        for stage in self.channel_space:
            local_c = []
            if isinstance(stage, list):
                for c in stage:
                    local_c.append(int(wm[i] * c))
                    i += 1
            else:
                local_c = int(wm[i] * stage)
            new_wm.append(local_c)

        real_ch = self.decode_ch(new_wm)
        return real_ch

    def set_real_ch(self, real_ch):
        for idx, stage_ch in enumerate(real_ch):
            self.features[idx].set_out_ch(stage_ch)
            if idx > 0:
                prev_ch = real_ch[idx-1] if isinstance(real_ch[idx-1], int) else real_ch[idx-1][-1]
                self.features[idx].set_in_ch(prev_ch)
        self.features[-4].set_in_ch(real_ch[-1][-1])

def slim_mobilenetv3_small(num_classes=1000):
    return SlimMobileNetV3(num_classes, mode='small')

def slim_mobilenetv3_large(num_classes=1000):
    return SlimMobileNetV3(num_classes, mode='large')

def slim_mobilenetv3_super(num_classes=1000):
    return SlimMobileNetV3(num_classes, mode='super')

def test():
    net = SlimMobileNetV3(mode='large')
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

    net = SlimMobileNetV3(mode='small')
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

    net = SlimMobileNetV3(mode='super')
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

if __name__ == '__main__':
    test()