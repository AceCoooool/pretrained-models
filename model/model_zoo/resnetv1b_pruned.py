"""Pruned ResNetV1bs, implemented in PyTorch."""
from __future__ import division
import json
import os
from collections import OrderedDict

import torch
from torch import nn

from model.model_zoo.resnet_v1b import ResNetV1b, BasicBlockV1b, BottleneckV1b

__all__ = ['resnet18_v1b_89', 'resnet50_v1d_86', 'resnet50_v1d_48', 'resnet50_v1d_37',
           'resnet50_v1d_11', 'resnet101_v1d_76', 'resnet101_v1d_73']


# TODO: this achievement is "ugly"
def prune_torch_block(net, params_name, params_shapes, params=None, pretrained=False):
    """
    :param params_shapes: dictionary of shapes of convolutional weights
    :param pretrained: Boolean specifying if the pretrained model parameters needs to be loaded
    :param net: original network that is required to be pruned
    :param params: dictionary of parameters for the pruned network. Size of the parameters in
    this dictionary tells what
    should be the size of channels of each convolution layer.
    :return: "net"
    """
    for layer in net.children():
        if isinstance(layer, nn.BatchNorm2d):
            layer.num_features = params_shapes[prune_torch_block.idx][0]
            if pretrained:
                param_name = params_name[prune_torch_block.idx]
                layer.weight.data = params[param_name + '.weight']
                layer.bias.data = params[param_name + '.bias']
                layer.running_mean = params[param_name + '.running_mean']
                layer.running_var = params[param_name + '.running_var']
            prune_torch_block.idx += 1

        if isinstance(layer, nn.Conv2d):
            param_shape = params_shapes[prune_torch_block.idx]
            layer.in_channels = param_shape[1]
            layer.out_channels = param_shape[0]
            if pretrained:
                param_name = params_name[prune_torch_block.idx]
                layer.weight.data = params[param_name + '.weight']
                if layer.bias is not None:
                    layer.bias.data = params[param_name + '.bias']
            prune_torch_block.idx += 1

        if isinstance(layer, nn.Linear):
            layer.in_features = params_shapes[prune_torch_block.idx][0]
            layer.out_features = params_shapes[prune_torch_block.idx][1]
            if pretrained:
                param_name = params_name[prune_torch_block.idx]
                layer.weight.data = params[param_name + '.weight']
                if layer.bias is not None:
                    layer.bias.data = params[param_name + '.bias']
            prune_torch_block.idx += 1
        else:
            prune_torch_block(layer, params_name, params_shapes, params, pretrained)


def resnet18_v1b_89(pretrained=None, **kwargs):
    """Constructs a ResNetV1b-18_2.6x model. Uses resnet18_v1b construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%db_%.1fx' % (18, 1, 2.6) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_86(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-50_1.8x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (50, 1, 1.8) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_48(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-50_3.6x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (50, 1, 3.6) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_37(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-50_5.9x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (50, 1, 5.9) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_11(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-50_8.8x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (50, 1, 8.8) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1d_76(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-101_1.9x model. Uses resnet101_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (101, 1, 1.9) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1d_73(pretrained=None, **kwargs):
    """Constructs a ResNetV1d-101_2.2x model. Uses resnet101_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : str
        load the default pretrained weights for model.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True,
                      **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'pruned_resnet/resnet%d_v%dd_%.1fx' % (101, 1, 2.2) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_items = json.load(jsonFile, object_pairs_hook=OrderedDict)
    prune_torch_block.idx = 0
    if pretrained:
        params = torch.load(pretrained)
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=params, pretrained=True)
    else:
        prune_torch_block(model, list(params_items.keys()), list(params_items.values()),
                          params=None, pretrained=False)
    if pretrained:
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


if __name__ == '__main__':
    net = resnet101_v1d_76()
    # print(net)
    name = list()
    for key in net.state_dict().keys():
        key = key.rsplit('.', 1)[0]
        if key not in name:
            name.append(key)
    for n in name:
        print("\"" + n + "\"")
    # print(net)
    # print(net.state_dict().keys())
    # print([v.shape for v in net.state_dict().values()])
    # for m in net.children():
    #     print(m)
