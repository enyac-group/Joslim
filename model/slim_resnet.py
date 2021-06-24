import torch
import torch.nn as nn

from .dyn_model import SlimBN, SlimConv2d, SlimDownsample, SlimDownsampleConv, SlimBasicBlock, SlimBottleneck, DynModel

__all__ = ['slim_resnet20', 'slim_resnet32', 'slim_resnet44', 'slim_resnet56',
           'slim_resnet18', 'slim_resnet34', 'slim_resnet50', 'slim_resnet101', 'slim_resnet152']

class SlimStem(nn.Module):
    def __init__(self, inC, outC, norm_layer):
        super(SlimStem, self).__init__()
        self.conv = SlimConv2d(3, outC, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = norm_layer(outC)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x

    def set_out_ch(self, ch):
        self.conv.set_out_ch(ch)
        self.norm.set_out_ch(ch)


class SlimStage(nn.Module):
    def __init__(self, N, block, inplanes, planes, stride=1, norm_layer=SlimBN, downsample_type=SlimDownsample):
        super(SlimStage, self).__init__()
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = downsample_type(inplanes, planes, stride)
        self.head = block(inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer)
        self.tail = []
        inplanes = planes * block.expansion
        for i in range(N - 1):
            self.tail.append(block(inplanes, planes, norm_layer=norm_layer))
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


class SlimResNet(DynModel):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=SlimBN):
        super(SlimResNet, self).__init__()
        num_stages = len(layers)

        self.block = block
        self._norm_layer = norm_layer

        self.og_channels = [16, 32, 64] if num_stages == 3 else [64, 128, 256, 512]
        self.channel_space = []
        for i in range(num_stages):
            unit = layers[i]+1
            if i == 0 and block == SlimBottleneck:
                c = [self.og_channels[i], self.og_channels[i]*block.expansion]
            else:
                c = [self.og_channels[i]*block.expansion]
            self.channel_space.append(c + [self.og_channels[i] for _ in range(unit-1)])

        self.layers = layers
        self.in_planes = self.og_channels[0]

        if num_stages == 3:
            # For CIFAR
            downsample_type = SlimDownsample
            stem = SlimConv2d(3, self.og_channels[0], kernel_size=1, bias=False)
        elif num_stages == 4:
            def helper(nIn, nOut, stride):
                layer = SlimDownsampleConv(nIn, nOut * block.expansion, stride, norm_layer)
                return layer
            downsample_type = helper
            # For ImageNet
            stem = SlimStem(3, self.og_channels[0], norm_layer)

        self.features = [stem]

        input_c = self.og_channels[0]
        for i in range(num_stages):
            stride = 1 if i == 0 else 2
            self.features.append(SlimStage(self.layers[i], block, input_c, self.og_channels[i],
                                           stride=stride, downsample_type=downsample_type))
            input_c = self.og_channels[i] * block.expansion

        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(SlimConv2d(self.og_channels[-1]*block.expansion, num_classes, kernel_size=1))
        self._initialize_weights(zero_init_residual)

        self.search_dim = sum([len(stage) for stage in self.channel_space])

    def _initialize_weights(self, zero_init_residual=False):
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.classifier[0].out_channels)
        return x

    def decode_ch(self, ch):
        prev_ch = 0
        real_ch = []
        for idx, stage_ch in enumerate(ch):
            expanded_stage_ch = stage_ch
            if idx > 0:
                stage_ch[0] = max(stage_ch[0], prev_ch)
                expanded_stage_ch = []
                for ci in range(len(stage_ch)-1):
                    expanded_stage_ch.append(stage_ch[ci+1])
                    expanded_stage_ch.append(stage_ch[0])
                prev_ch = stage_ch[0]
            real_ch.append(expanded_stage_ch)
        return real_ch

    def decode_wm(self, wm):
        new_wm = [int(wm[0]*self.channel_space[0][0])]
        i = 0
        for si, stage in enumerate(self.channel_space):
            local_c = []
            for ci, c in enumerate(stage):
                if not (si == 0 and ci == 0 and self.block == SlimBottleneck):
                    local_c.append(int(wm[i] * c))
                i += 1
            new_wm.append(local_c)

        real_ch = self.decode_ch(new_wm)
        return real_ch

    def set_real_ch(self, real_ch):
        for idx, stage_ch in enumerate(real_ch):
            self.features[idx].set_out_ch(stage_ch)
            if idx > 0:
                prev_ch = real_ch[idx-1] if isinstance(real_ch[idx-1], int) else real_ch[idx-1][-1]
                self.features[idx].set_in_ch(prev_ch)
        self.classifier[0].set_in_ch(real_ch[-1][-1])


# For CIFAR
def slim_resnet20(num_classes=10):
    return SlimResNet(SlimBasicBlock, [3,3,3], num_classes=num_classes)

def slim_resnet32(num_classes=10):
    return SlimResNet(SlimBasicBlock, [5,5,5], num_classes=num_classes)

def slim_resnet44(num_classes=10):
    return SlimResNet(SlimBasicBlock, [7,7,7], num_classes=num_classes)

def slim_resnet56(num_classes=10):
    return SlimResNet(SlimBasicBlock, [9,9,9], num_classes=num_classes)

# For ImageNet
def slim_resnet18(num_classes=1000):
    return SlimResNet(SlimBasicBlock, [2,2,2,2], num_classes=num_classes)

def slim_resnet34(num_classes=1000):
    return SlimResNet(SlimBasicBlock, [3,4,6,3], num_classes=num_classes)

def slim_resnet50(num_classes=1000):
    return SlimResNet(SlimBottleneck, [3,4,6,3], num_classes=num_classes)

def slim_resnet101(num_classes=1000):
    return SlimResNet(SlimBottleneck, [3,4,23,3], num_classes=num_classes)

def slim_resnet152(num_classes=1000):
    return SlimResNet(SlimBottleneck, [3,8,36,3], num_classes=num_classes)



def test():
    net = slim_resnet56()
    print(net.search_dim)
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

    net = slim_resnet18()
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

    net = slim_resnet50()
    print(net.search_dim)
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    out_ch = net.random_sample_wm()
    print(out_ch)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

if __name__ == '__main__':
    test()

