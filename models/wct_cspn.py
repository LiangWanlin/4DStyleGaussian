0#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:32:49 2018

@author: norbot
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
# import libs.cspn as post_process
import models.cspn as post_process
from torch.autograd import Variable
# import libs.update_model
import models.wct_update_model as update_model
import torch.nn.functional as F

# memory analyze
import gc
#torch.manual_seed(4) 

__all__ = ['ResNet', 'resnet50']


model_path ={
    'resnet18': 'pretrained/resnet18.pth',
    'resnet50': 'pretrained/resnet50.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        oheight = 0
        owidth = 0
        if self.oheight == 0 and self.owidth == 0:
            oheight = scale * x.size(2)
            owidth = scale * x.size(3)
            x = self._up_pool(x)
        else:
            oheight = self.oheight
            owidth = self.owidth
            x = self._up_pool(x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Simple_Gudi_UpConv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)


    def _up_pooling(self, x, scale):

        x = self._up_pool(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x.narrow(2,0,self.oheight)
            x = x.narrow(3,0,self.owidth)
        return x


    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        return out


class Simple_Gudi_UpConv_Block_Last_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        
        _,_,height,width = x.size()
        height = height*2
        width = width*2

        x = self._up_pool(x)
        if height != 0 and width != 0:
            x = x.narrow(2, 0, height)
            x = x.narrow(3, 0, width)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out


class Gudi_UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth

    def _up_pooling(self, x, scale):
        
        _,_,height,width = x.size()
        height = height*2
        width = width*2
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if height !=0 and width !=0:
            x = x[:,:,0:height, 0:width]
        mask = torch.zeros_like(x)
        for h in range(0, height, 2):
            for w in range(0, width, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        self._up_pool = Unpool(in_channels)

    def _up_pooling(self, x, scale):
        
        _,_,height,width = x.size()
        height = height*2
        width = width*2

        x = self._up_pool(x)
        if height !=0 and width !=0:
            x = x.narrow(2, 0, height)
            x = x.narrow(3, 0, width)
        return x

    def forward(self, x, side_input):
        #print("x.shape",x.shape)
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        #print("out.shape",out.shape)
        #print("side_input.shape",side_input.shape)
        out = torch.cat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, up_proj_block, cspn_config=None):
        self.inplanes = 64
        cspn_config_default = {'step': 24, 'kernel': 3, 'norm_type': '8sum'}
        if not (cspn_config is None):
            cspn_config_default.update(cspn_config)
        print(cspn_config_default)

        super(ResNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mid_channel = 256*block.expansion
        self.conv2 = nn.Conv2d(512*block.expansion, 512*block.expansion, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512*block.expansion)
        self.up_proj_layer1 = self._make_up_conv_layer(up_proj_block,
                                                       self.mid_channel,
                                                       int(self.mid_channel/2))
        self.up_proj_layer2 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel/2),
                                                       int(self.mid_channel/4))
        self.up_proj_layer3 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel/4),
                                                       int(self.mid_channel/8))
        self.up_proj_layer4 = self._make_up_conv_layer(up_proj_block,
                                                       int(self.mid_channel/8),
                                                       int(self.mid_channel/16))
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.post_process_layer = self._make_post_process_layer(cspn_config_default)
        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 16,16)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 32,32)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 64,64)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 128,128)
#        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 256, 256)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 8, 256, 256)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)

    def _make_post_process_layer(self, cspn_config=None):
        return post_process.Affinity_Propagate(cspn_config['step'],
                                               cspn_config['kernel'],
                                               norm_type=cspn_config['norm_type'])

    def forward(self, x, distorted):
        [batch_size, channel, height, width] = x.size()
#        sparse_depth = x.narrow(1,3,1).clone()
        x = self.conv1_1(x)
        #print("x.shape",x.shape)
        skip4 = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(self.conv2(x))

        x = self.gud_up_proj_layer1(x)
        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)

        guidance = self.gud_up_proj_layer6(x)
#        x= self.gud_up_proj_layer5(x)

        prop = self.post_process_layer(guidance, distorted)
        return prop


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model from ', model_path['resnet50'])
        pretrained_dict = torch.load(model_path['resnet50'])
        model.load_state_dict(update_model.update_model(model, pretrained_dict))
    return model



