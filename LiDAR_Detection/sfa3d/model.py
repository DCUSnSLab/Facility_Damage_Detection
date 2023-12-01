from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import argparse
# from torchsummary import summary
from easydict import EasyDict as edict


class fpn_resnet:
    BN_MOMENTUM = 0.1

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(fpn_resnet.BasicBlock, self).__init__()
            self.conv1 = fpn_resnet.conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes, momentum=fpn_resnet.BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = fpn_resnet.conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes, momentum=fpn_resnet.BN_MOMENTUM)
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
            super(fpn_resnet.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, momentum=fpn_resnet.BN_MOMENTUM)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, momentum=fpn_resnet.BN_MOMENTUM)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=fpn_resnet.BN_MOMENTUM)
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

    class PoseResNet(nn.Module):

        def __init__(self, block, layers, heads, head_conv, **kwargs):
            self.inplanes = 64
            self.deconv_with_bias = False
            self.heads = heads

            super(fpn_resnet.PoseResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=fpn_resnet.BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
            self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
            self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

            fpn_channels = [256, 128, 64]
            for fpn_idx, fpn_c in enumerate(fpn_channels):
                for head in sorted(self.heads):
                    num_output = self.heads[head]
                    if head_conv > 0:
                        fc = nn.Sequential(
                            nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
                    else:
                        fc = nn.Conv2d(in_channels=fpn_c, out_channels=num_output, kernel_size=1, stride=1, padding=0)

                    self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=fpn_resnet.BN_MOMENTUM),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            _, _, input_h, input_w = x.size()
            hm_h, hm_w = input_h // 4, input_w // 4
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            out_layer1 = self.layer1(x)
            out_layer2 = self.layer2(out_layer1)

            out_layer3 = self.layer3(out_layer2)

            out_layer4 = self.layer4(out_layer3)

            # up_level1: torch.Size([b, 512, 14, 14])
            up_level1 = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)

            concat_level1 = torch.cat((up_level1, out_layer3), dim=1)
            # up_level2: torch.Size([b, 256, 28, 28])
            up_level2 = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear',
                                      align_corners=True)

            concat_level2 = torch.cat((up_level2, out_layer2), dim=1)
            # up_level3: torch.Size([b, 128, 56, 56]),
            up_level3 = F.interpolate(self.conv_up_level2(concat_level2), scale_factor=2, mode='bilinear',
                                      align_corners=True)
            # up_level4: torch.Size([b, 64, 56, 56])
            up_level4 = self.conv_up_level3(torch.cat((up_level3, out_layer1), dim=1))

            ret = {}
            for head in self.heads:
                temp_outs = []
                for fpn_idx, fdn_input in enumerate([up_level2, up_level3, up_level4]):
                    fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                    _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                    # Make sure the added features having same size of heatmap output
                    if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                        fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                    temp_outs.append(fpn_out)
                # Take the softmax in the keypoint feature pyramid network
                final_out = self.apply_kfpn(temp_outs)

                ret[head] = final_out

            return ret

        def apply_kfpn(self, outs):
            outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
            softmax_outs = F.softmax(outs, dim=-1)
            ret_outs = (outs * softmax_outs).sum(dim=-1)
            return ret_outs

        def init_weights(self, num_layers, pretrained=True):
            if pretrained:
                # TODO: Check initial weights for head later
                for fpn_idx in [0, 1, 2]:  # 3 FPN layers
                    for head in self.heads:
                        final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                        for i, m in enumerate(final_layer.modules()):
                            if isinstance(m, nn.Conv2d):
                                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                                # print('=> init {}.bias as 0'.format(name))
                                if m.weight.shape[0] == self.heads[head]:
                                    if 'hm' in head:
                                        nn.init.constant_(m.bias, -2.19)
                                    else:
                                        nn.init.normal_(m.weight, std=0.001)
                                        nn.init.constant_(m.bias, 0)
                # pretrained_state_dict = torch.load(pretrained)
                url = fpn_resnet.model_urls['resnet{}'.format(num_layers)]
                pretrained_state_dict = model_zoo.load_url(url)
                print('=> loading pretrained model {}'.format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                   34: (BasicBlock, [3, 4, 6, 3]),
                   50: (Bottleneck, [3, 4, 6, 3]),
                   101: (Bottleneck, [3, 4, 23, 3]),
                   152: (Bottleneck, [3, 8, 36, 3])}

    def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):
        block_class, layers = fpn_resnet.resnet_spec[num_layers]

        model = fpn_resnet.PoseResNet(block_class, layers, heads, head_conv=head_conv)
        model.init_weights(num_layers, pretrained=imagenet_pretrained)
        return model

class model_utils:

    def create_model(self):
        """Create model based on architecture name"""
        try:
            arch_parts = self.arch.split('_')
            num_layers = int(arch_parts[-1])
        except:
            raise ValueError
        if 'fpn_resnet' in self.arch:
            print('using ResNet architecture with feature pyramid')
            model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=self.heads, head_conv=self.head_conv,
                                            imagenet_pretrained=self.imagenet_pretrained)
        elif 'resnet' in self.arch:
            print('using ResNet architecture')
            model = resnet.get_pose_net(num_layers=num_layers, heads=self.heads, head_conv=self.head_conv,
                                        imagenet_pretrained=self.imagenet_pretrained)
        else:
            assert False, 'Undefined model backbone'

        return model

    def get_num_parameters(model):
        """Count number of trained parameters of the model"""
        if hasattr(model, 'module'):
            num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        else:
            num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return num_parameters

    def make_data_parallel(model, configs):
        if configs.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if configs.gpu_idx is not None:
                torch.cuda.set_device(configs.gpu_idx)
                model.cuda(configs.gpu_idx)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
                configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model = model.cuda(configs.gpu_idx)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

        return model
    def main(self):
        parser = argparse.ArgumentParser(description='RTM3D Implementation')
        parser.add_argument('-a', '--arch', type=str, default='resnet_18', metavar='ARCH',
                            help='The name of the model architecture')
        parser.add_argument('--head_conv', type=int, default=-1,
                            help='conv layer channels for output head'
                                 '0 for no conv layer'
                                 '-1 for default setting: '
                                 '64 for resnets and 256 for dla.')

        configs = edict(vars(parser.parse_args()))
        if configs.head_conv == -1:  # init default head_conv
            configs.head_conv = 256 if 'dla' in configs.arch else 64

        configs.num_classes = 8
        configs.num_vertexes = 8
        configs.num_center_offset = 2
        configs.num_vertexes_offset = 2
        configs.num_dimension = 3
        configs.num_rot = 8
        configs.num_depth = 1
        configs.num_wh = 2
        configs.heads = {
            'hm_mc': configs.num_classes,
            'hm_ver': configs.num_vertexes,
            'vercoor': configs.num_vertexes * 2,
            'cenoff': configs.num_center_offset,
            'veroff': configs.num_vertexes_offset,
            'dim': configs.num_dimension,
            'rot': configs.num_rot,
            'depth': configs.num_depth,
            'wh': configs.num_wh
        }

        configs.device = torch.device('cuda:1')
        # configs.device = torch.device('cpu')

        model = model_utils.create_model(configs).to(device=configs.device)
        sample_input = torch.randn((1, 3, 224, 224)).to(device=configs.device)
        # summary(model.cuda(1), (3, 224, 224))
        output = model(sample_input)
        for hm_name, hm_out in output.items():
            print('hm_name: {}, hm_out size: {}'.format(hm_name, hm_out.size()))

        print('number of parameters: {}'.format(model_utils.get_num_parameters(model)))
class resnet:
    BN_MOMENTUM = 0.1

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(resnet.BasicBlock, self).__init__()
            self.conv1 = resnet.conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes, momentum=resnet.BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = resnet.conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes, momentum=resnet.BN_MOMENTUM)
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
            super(resnet.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, momentum=resnet.BN_MOMENTUM)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, momentum=resnet.BN_MOMENTUM)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                      momentum=resnet.BN_MOMENTUM)
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

    class PoseResNet(nn.Module):

        def __init__(self, block, layers, heads, head_conv, **kwargs):
            self.inplanes = 64
            self.deconv_with_bias = False
            self.heads = heads

            super(resnet.PoseResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=resnet.BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            # used for deconv layers
            self.deconv_layers = self._make_deconv_layer(
                3,
                [256, 256, 256],
                [4, 4, 4],
            )
            # self.final_layer = []

            for head in sorted(self.heads):
                num_output = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(256, head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output,
                                  kernel_size=1, stride=1, padding=0))
                else:
                    fc = nn.Conv2d(
                        in_channels=256,
                        out_channels=num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                self.__setattr__(head, fc)

            # self.final_layer = nn.ModuleList(self.final_layer)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=resnet.BN_MOMENTUM),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def _get_deconv_cfg(self, deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
            assert num_layers == len(num_filters), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'
            assert num_layers == len(num_kernels), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

                planes = num_filters[i]
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=self.inplanes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias))
                layers.append(nn.BatchNorm2d(planes, momentum=resnet.BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))
                self.inplanes = planes

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.deconv_layers(x)
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(x)
            return ret

        def init_weights(self, num_layers, pretrained=True):
            if pretrained:
                # print('=> init resnet deconv weights from normal distribution')
                for _, m in self.deconv_layers.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        nn.init.normal_(m.weight, std=0.001)
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        # print('=> init {}.weight as 1'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                # print('=> init final conv weights from normal distribution')
                for head in self.heads:
                    final_layer = self.__getattr__(head)
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                            # print('=> init {}.bias as 0'.format(name))
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)
                # pretrained_state_dict = torch.load(pretrained)
                url = resnet.model_urls['resnet{}'.format(num_layers)]
                pretrained_state_dict = model_zoo.load_url(url)
                print('=> loading pretrained model {}'.format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                   34: (BasicBlock, [3, 4, 6, 3]),
                   50: (Bottleneck, [3, 4, 6, 3]),
                   101: (Bottleneck, [3, 4, 23, 3]),
                   152: (Bottleneck, [3, 8, 36, 3])}

    def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):
        block_class, layers = resnet.resnet_spec[num_layers]

        model = resnet.PoseResNet(block_class, layers, heads, head_conv=head_conv)
        model.init_weights(num_layers, pretrained=imagenet_pretrained)
        return model
