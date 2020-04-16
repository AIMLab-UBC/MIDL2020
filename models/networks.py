from torch import nn
import torchvision
import torch


class Baseline(nn.Module):
    def __init__(self, num_classes=5, use_pretrained=False):
        super(Baseline, self).__init__()
        # init vgg19_bn
        self.cnn = torchvision.models.vgg19_bn(pretrained=use_pretrained)
        # modify the last fully-connected layer
        self.cnn.classifier._modules['6'] = nn.Linear(4096, num_classes)

    def forward(self, x):
        # forward pass to compute logits
        logits = self.cnn(x)
        return logits


class TwoStageCNN(nn.Module):
    def _initialize_weights(self, modules):
        # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _add_conv_module(self, features, slice_idx=3):
        add_module = nn.ModuleList()
        # attach two new conv blocks
        add_sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self._initialize_weights(add_sequential)
        # append the new conv blocks
        for layer in add_sequential:
            add_module.append(layer)
        # remove the first conv block
        for layer in features[slice_idx:]:
            add_module.append(layer)
        return nn.Sequential(*add_module)

    def __init__(self, num_classes=5, use_pretrained=False, patch_size=256, weights_save_path=None):
        super(TwoStageCNN, self).__init__()
        # backbone model
        model = torchvision.models.vgg19_bn(pretrained=use_pretrained)
        # modify the last fully-connected layer
        model.classifier._modules['6'] = nn.Linear(4096, num_classes)
        # load weights from 256 * 256
        # and change model for 512 * 512 input
        if patch_size == 512:
            # load stored state
            state = torch.load(weights_save_path)
            # load previous size learned weights and discard first conv block then add new conv blocks
            model.load_state_dict(state['state_dict'])
            model.features = self._add_conv_module(model.features)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
