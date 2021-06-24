import torch
import torch.nn as nn
import math

from .dyn_model import SlimBN, SlimConv2d, SlimDWSConv, SlimInvertedResidual, DynModel

__all__ = ['slim_mobilenetv2']

class SlimConvBN(nn.Module):
    def __init__(self, inC, outC, kernel_size, stride):
        super(SlimConvBN, self).__init__()
        self.conv = SlimConv2d(inC, outC, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.norm = SlimBN(outC)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

    def set_out_ch(self, ch):
        self.conv.set_out_ch(ch)
        self.norm.set_out_ch(ch)

    def set_in_ch(self, ch):
        self.conv.set_in_ch(ch)


class SlimStage(nn.Module):
    def __init__(self, N, input_channel, output_channel, stride=1, expand_ratio=6):
        super(SlimStage, self).__init__()
        self.head = SlimInvertedResidual(input_channel, output_channel, stride, expand_ratio*input_channel)
        self.tail = []
        input_channel = output_channel
        for i in range(N - 1):
            self.tail.append(SlimInvertedResidual(input_channel, output_channel, 1, expand_ratio*input_channel))
            input_channel = output_channel
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


class SlimMobileNetV2(DynModel):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(SlimMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]


        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [SlimConvBN(3, input_channel, kernel_size=3, stride=2)]
        self.features.append(SlimDWSConv(input_channel, int(16*width_mult), 1))
        self.channel_space = [input_channel, [int(16*width_mult)]]
        input_channel = int(16*width_mult)
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            channel = [output_channel, input_channel * t] # first component for residual
            self.features.append(SlimStage(n, input_channel, output_channel, s, t))
            channel += [output_channel*t for _ in range(n-1)]
            input_channel = output_channel
            self.channel_space.append(channel)
        # building last several layers
        self.features.append(SlimConvBN(input_channel, self.last_channel, kernel_size=1, stride=1))
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

        self.search_dim = 2 + sum([n for t, c, n, s in self.interverted_residual_setting]) + len(self.interverted_residual_setting)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[1].in_features)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels / m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

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
        self.features[-2].set_in_ch(real_ch[-1][-1])

def slim_mobilenetv2(num_classes=1000):
    return SlimMobileNetV2(num_classes)

def test():
    net = SlimMobileNetV2()
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

if __name__ == '__main__':
    test()
