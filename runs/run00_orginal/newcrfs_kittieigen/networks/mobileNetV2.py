import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from torchvision import models

class deepFeatureExtractor_MobileNetV2(nn.Module):
    def __init__(self):
        super(deepFeatureExtractor_MobileNetV2, self).__init__()

        # after passing 2th : H/4  x W/4
        # after passing 3th : H/8  x W/8
        # after passing 4th : H/16 x W/16
        # after passing 5th : H/32 x W/32
        self.encoder = models.mobilenet_v2(pretrained=True)
        del self.encoder.classifier
        self.layerList = [3, 6, 13, 17]
        self.dimList = [24, 32, 96, 320]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return out_featList

    # 未使用
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
