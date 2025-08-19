import torch
import torch.nn as nn


from torchvision import models

class deepFeatureExtractor_MobileNetV2(nn.Module):  # 1.81M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_MobileNetV2, self).__init__()

        # after passing 2th : H/4  x W/4
        # after passing 3th : H/8  x W/8
        # after passing 4th : H/16 x W/16
        # after passing 5th : H/32 x W/32
        self.encoder = models.mobilenet_v2(pretrained=pretrained)
        del self.encoder.classifier
        del self.encoder.features[-1]
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
        return tuple(out_featList)

    # # 未使用
    # def freeze_bn(self, enable=False):
    #     """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
    #     for module in self.modules():
    #         if isinstance(module, nn.BatchNorm2d):
    #             module.train() if enable else module.eval()

    #             module.weight.requires_grad = enable
    #             module.bias.requires_grad = enable


class deepFeatureExtractor_ResNet18(nn.Module):  # 11.18M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_ResNet18, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.resnet18(pretrained=pretrained)

        del self.encoder.fc
        self.layerList = ['layer1','layer2','layer3', 'layer4']
        self.dimList = [64, 128, 256, 512]
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 

        return tuple(out_featList)
    

class deepFeatureExtractor_EfficientNetB0(nn.Module):  # 3.60M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_EfficientNetB0, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.efficientnet_b0(pretrained=pretrained)

        del self.encoder.classifier
        del self.encoder.features[-1]
        self.layerList = [2, 3, 5, 7]
        self.dimList = [24, 40, 112, 320]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)
    

class deepFeatureExtractor_EfficientNetB4(nn.Module):  # 16.74M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_EfficientNetB4, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.efficientnet_b4(pretrained=pretrained)
        del self.encoder.classifier
        del self.encoder.features[-1]
        self.layerList = [2, 3, 5, 7]
        self.dimList = [32, 56, 160, 448]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_EfficientNetB7_2(nn.Module):  # 12.32M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_EfficientNetB7_2, self).__init__()

        # after passing Layer0 : H/2 x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.efficientnet_b7(pretrained=pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        del self.encoder.classifier
        del self.encoder.features[-3:]
        self.layerList = [1, 2, 3, 5]
        self.dimList = [32, 48, 80, 224]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)    

class deepFeatureExtractor_MNASNet1_0(nn.Module):  # 2.69M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_MNASNet1_0, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.mnasnet1_0(pretrained=pretrained)

        del self.encoder.classifier
        del self.encoder.layers[-3:]  # 删除最后3层
        self.layerList = [8, 9, 11, 13]
        self.dimList = [24, 40, 96, 320]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.layers)):
            feature = self.encoder.layers[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_MNASNet0_75(nn.Module):  # 1.58M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_MNASNet0_75, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.mnasnet0_75(pretrained=pretrained)

        del self.encoder.classifier
        del self.encoder.layers[-3:]  # 删除最后3层
        self.layerList = [8, 9, 11, 13]
        self.dimList = [24, 32, 72, 240]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.layers)):
            feature = self.encoder.layers[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_ShuffleNet_V2(nn.Module):  # 1.25M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_ShuffleNet_V2, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.shufflenet_v2_x1_0(pretrained=pretrained)

        del self.encoder.fc
        del self.encoder.conv5
        self.layerList = ['maxpool','stage2','stage3', 'stage4']
        self.dimList = [24, 116, 232, 464]

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():

            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            # print(feature.shape)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 
        # assert False
        return tuple(out_featList)
    

class deepFeatureExtractor_Mobilenet_v3_small(nn.Module):  # 0.93M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_Mobilenet_v3_small, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.mobilenet_v3_small(pretrained=pretrained)

        del self.encoder.classifier
        del self.encoder.avgpool
        # del self.encoder.features[-3:]  # 删除最后3层
        self.layerList = [1, 3, 8, 11]
        self.dimList = [16, 24, 48, 96]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_Mobilenet_v3_large(nn.Module):  # 2.97M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_Mobilenet_v3_large, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.mobilenet_v3_large(pretrained=pretrained)

        del self.encoder.classifier
        del self.encoder.avgpool
        # del self.encoder.features[-3:]  # 删除最后3层
        self.layerList = [3, 6, 12, 15]
        self.dimList = [24, 40, 112, 160]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)
    

class deepFeatureExtractor_RegNet_Y_400MF(nn.Module):  # 3.90M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_RegNet_Y_400MF, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.regnet_y_400mf(pretrained=pretrained)

        del self.encoder.fc
        self.layerList = [0, 1, 2, 3]
        self.dimList = [48, 104, 208, 440]

    def forward(self, x):
        out_featList = []
        # feature = x
        feature = self.encoder.stem(x)
        for i in range(len(self.encoder.trunk_output)):
            feature = self.encoder.trunk_output[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_RegNet_X_400MF(nn.Module):  # 5.09M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_RegNet_X_400MF, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.regnet_x_400mf(pretrained=pretrained)

        del self.encoder.fc
        self.layerList = [0, 1, 2, 3]
        self.dimList = [32, 64, 160, 440]

    def forward(self, x):
        out_featList = []
        # feature = x
        feature = self.encoder.stem(x)
        for i in range(len(self.encoder.trunk_output)):
            feature = self.encoder.trunk_output[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_RegNet_X_8gF(nn.Module):  # 37.65M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_RegNet_X_8gF, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.regnet_x_8gf(pretrained=pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)

        del self.encoder.fc
        self.layerList = [0, 1, 2, 3]
        self.dimList = [80, 240, 720, 1920]

    def forward(self, x):
        out_featList = []
        feature = self.encoder.stem(x)
        for i in range(len(self.encoder.trunk_output)):
            feature = self.encoder.trunk_output[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_RegNet_X_8gF_2(nn.Module):  # 29.11M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_RegNet_X_8gF_2, self).__init__()
        # after passing Layer0 : H/2 x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.regnet_x_8gf(pretrained=pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)

        del self.encoder.trunk_output[3]
        del self.encoder.fc
        self.layerList = [0, 1, 2]
        self.dimList = [32, 80, 240, 720]

    def forward(self, x):
        out_featList = []
        feature = x
        feature = self.encoder.stem(feature)
        out_featList.append(feature)
        for i in range(len(self.encoder.trunk_output)):
            feature = self.encoder.trunk_output[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)
    

class deepFeatureExtractor_ConvNeXt_Tiny(nn.Module):  # 27.82M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_ConvNeXt_Tiny, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.convnext_tiny(pretrained=pretrained)

        del self.encoder.classifier
        self.layerList = [1, 3, 5, 7]
        self.dimList = [96, 192, 384, 768]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_DenseNet121(nn.Module):  # 6.95M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_DenseNet121, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.densenet121(pretrained=pretrained)

        del self.encoder.classifier
        self.layerList = [4, 6, 8, 11]
        self.dimList = [256, 512, 1024, 1024]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_DenseNet161(nn.Module):  # 26.47M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_DenseNet161, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.densenet161(pretrained=pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        del self.encoder.features[-1:]
        del self.encoder.classifier
        self.layerList = [4, 6, 8, 10]
        self.dimList = [384, 768, 2112, 2208]

    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            # print(i, feature.shape)
            if i in self.layerList:
                out_featList.append(feature)
        # assert False
        return tuple(out_featList)

class deepFeatureExtractor_DenseNet161_2(nn.Module):  # 16.98M
    def __init__(self, pretrained=False):  # load pretrained weight
        super(deepFeatureExtractor_DenseNet161_2, self).__init__()

        # after passing Layer0 : H/2 x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.densenet161(pretrained=pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        print('!!!backbone pretrained weight:', pretrained)
        
        del self.encoder.classifier
        del self.encoder.features.norm5
        del self.encoder.features.denseblock4

        self.dimList = [96, 192, 384, 1056]
        

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder.features._modules.items():
            if ('transition' in k):
                feature = v.norm(feature)
                feature = v.relu(feature)
                feature = v.conv(feature)
                out_featList.append(feature)
                feature = v.pool(feature)
            elif k == 'conv0':
                feature = v(feature)
                out_featList.append(feature)
            else:
                feature = v(feature)
        return tuple(out_featList)

class deepFeatureExtractor_ResNeXt50_32x4d(nn.Module):  # 22.98M
    def __init__(self, pretrained=True):  # load pretrained weight
        super(deepFeatureExtractor_ResNeXt50_32x4d, self).__init__()

        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        # after passing Layer4 : H/32 x W/32
        self.encoder = models.resnext50_32x4d(pretrained=pretrained)

        del self.encoder.fc
        self.layerList = ['layer1','layer2','layer3', 'layer4']
        self.dimList = [256, 512, 1024, 2048]
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 

        return tuple(out_featList)
    
# class deepFeatureExtractor_SqueezeNet1_1(nn.Module):  # 0.72M 尺寸不规整
#     def __init__(self, pretrained=True):  # load pretrained weight
#         super(deepFeatureExtractor_SqueezeNet1_1, self).__init__()

#         # after passing Layer1 : H/4  x W/4
#         # after passing Layer2 : H/8  x W/8
#         # after passing Layer3 : H/16 x W/16
#         # after passing Layer4 : H/32 x W/32
#         self.encoder = models.squeezenet1_1(pretrained=pretrained)

#         del self.encoder.classifier
#         # del self.encoder.features[-3:]  # 删除最后3层
#         self.layerList = [8, 9, 11, 13]
#         self.dimList = [24, 40, 96, 320]

#     def forward(self, x):
#         out_featList = []
#         feature = x
#         for i in range(len(self.encoder.features)):
#             feature = self.encoder.features[i](feature)
#             print(i, feature.shape)
#             if i in self.layerList:
#                 out_featList.append(feature)
#         # assert False
#         return tuple(out_featList)


    

def main():
    import numpy as np
    # Initialize the model
    model = deepFeatureExtractor_EfficientNetB4(pretrained=False).cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {:.2f}M".format(num_params/1e6))

    # Convert numpy array to torch tensor and move it to GPU
    input = np.random.random((1,3,352,1216))
    input_tensor = torch.from_numpy(input).float().cuda()

    # Pass the tensor through the model
    outputs = model(input_tensor)

    # Print the output shapes
    for output in outputs:
        print(output.shape)

if __name__ == "__main__":
    main()