import torch.nn as nn
from same_padding import Conv2d
class _ConvLayer(nn.Sequential):
    def __init__(self, input_features, output_features, kernel_size, stride):
        super(_ConvLayer, self).__init__()
        self.add_module('conv2', Conv2d(in_channels=input_features, out_channels=output_features,
                                        kernel_size=kernel_size, stride=stride))
        self.add_module('relu', nn.ReLU(inplace=True))
class _ConvLayer_sigmoid(nn.Sequential):
    def __init__(self, input_features, output_features, kernel_size, stride):
        super(_ConvLayer_sigmoid, self).__init__()
        self.add_module('conv2', Conv2d(in_channels=input_features, out_channels=output_features,
                                        kernel_size=kernel_size, stride=stride))
        self.add_module('sigmoid', nn.Sigmoid())

class _Conv_Unsampling(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, stride, scale_factor):
        super(_Conv_Unsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(_ConvLayer(input_features, output_features, kernel_size, stride))
    def forward(self, x):
        x = self.conv(x)
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


class AEModel(nn.Module):
    def __init__(self, color_mode):
        super(AEModel, self).__init__()

        if(color_mode == 'rgb'):
            channel = 3
        else:
            channel = 1

        self.encoder = nn.Sequential(
            _ConvLayer(channel, 32, 4, 2),
            _ConvLayer(32, 32, 4, 2),
            _ConvLayer(32, 32, 4, 2),
            _ConvLayer(32, 32, 3, 1),
            _ConvLayer(32, 64, 4, 2),
            _ConvLayer(64, 64, 3, 1),
            _ConvLayer(64, 128, 4, 2),
            _ConvLayer(128, 64, 3, 1),
            _ConvLayer(64, 32, 3, 1),
            _ConvLayer(32, 1, 8, 1),
        )

        self.decoder1 = _ConvLayer(1, 32, 3, 1)
        self.decoder2 = nn.Sequential(
            _Conv_Unsampling(32, 64, 3, 1, 2),
            _Conv_Unsampling(64, 128, 4, 2, 2),
            _Conv_Unsampling(128, 64, 3, 1, 2),
            _Conv_Unsampling(64, 64, 4, 2, 2),
            _Conv_Unsampling(64, 32, 3, 1, 2),
            _Conv_Unsampling(32, 32, 4, 2, 4),
            _Conv_Unsampling(32, 32, 4, 2, 2),
            _Conv_Unsampling(32, 32, 8, 1, 2),

        )
        self.decoder3 = _ConvLayer_sigmoid(32, channel, 8, 1)
    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x
