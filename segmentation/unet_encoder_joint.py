import torch.nn as nn

from model_utils import *
import sys
sys.path.insert(1, "../pdr")
import pdr

class unet_encoder(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_encoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        #print([conv1.shape, conv2.shape, conv3.shape, conv4.shape, center.shape])
        #exit()
        return conv1, conv2, conv3, conv4, center


class unet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.device = torch.device("cuda")
        self.encoder1 = unet_encoder(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm)
        self.encoder1 = self.encoder1.to(self.device)
        ckpt_path = "pdr.ckpt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        checkpoint = {}
        for key, val in ckpt['netD'].items():
            if 'network_down' in key:
                key = key[13:]
                checkpoint[key] = val
        self.encoder1.load_state_dict(checkpoint)
        checkpoint = {}
        for key, val in ckpt['netA'].items():
            if 'network_down' in key:
                key = key[13:]
                checkpoint[key] = val
        self.encoder2 = unet_encoder(feature_scale, n_classes, is_deconv, in_channels, is_batchnorm)
        self.encoder2 = self.encoder2.to(self.device)
        self.encoder2.load_state_dict(checkpoint)

        filters = [2 * val for val in filters]
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1_1, conv1_2, conv1_3, conv1_4, center_1 = self.encoder1(inputs)
        conv2_1, conv2_2, conv2_3, conv2_4, center_2 = self.encoder2(inputs)
        conv1 = torch.cat([conv1_1,conv2_1],dim=1)
        conv2 = torch.cat([conv1_2,conv2_2],dim=1)
        conv3 = torch.cat([conv1_3,conv2_3],dim=1)
        conv4 = torch.cat([conv1_4,conv2_4],dim=1)
        center = torch.cat([center_1,center_1],dim=1)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
