from .model_zoo.resnet import *
from .model_zoo.resnet_v1b import *
from .model_zoo.mobilenet import *
from .model_zoo.resnext import *
from .model_zoo.vgg import *
from .model_zoo.resnetv1b_pruned import *

_models = {
    # -----------------------------------------------------------------------------
    # resnet
    # -----------------------------------------------------------------------------
    'resnet18_v1': resnet18_v1,
    'resnet34_v1': resnet34_v1,
    'resnet50_v1': resnet50_v1,
    'resnet101_v1': resnet101_v1,
    'resnet152_v1': resnet152_v1,
    'resnet18_v2': resnet18_v2,
    'resnet34_v2': resnet34_v2,
    'resnet50_v2': resnet50_v2,
    'resnet101_v2': resnet101_v2,
    'resnet152_v2': resnet152_v2,
    # -----------------------------------------------------------------------------
    # resnet_v1b
    # -----------------------------------------------------------------------------
    'resnet18_v1b': resnet18_v1b,
    'resnet34_v1b': resnet34_v1b,
    'resnet50_v1b': resnet50_v1b,
    'resnet50_v1b_gn': resnet50_v1b_gn,
    'resnet101_v1b': resnet101_v1b,
    'resnet101_v1b_gn': resnet101_v1b_gn,
    'resnet152_v1b': resnet152_v1b,
    'resnet50_v1c': resnet50_v1c,
    'resnet101_v1c': resnet101_v1c,
    'resnet152_v1c': resnet152_v1c,
    'resnet50_v1d': resnet50_v1d,
    'resnet101_v1d': resnet101_v1d,
    'resnet152_v1d': resnet152_v1d,
    # -----------------------------------------------------------------------------
    # resnext
    # -----------------------------------------------------------------------------
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'se_resnext101_64x4d': se_resnext101_64x4d,
    # -----------------------------------------------------------------------------
    # mobilenet
    # -----------------------------------------------------------------------------
    'mobilenet1.0': mobilenet1_0,
    'mobilenet0.75': mobilenet0_75,
    'mobilenet0.5': mobilenet0_5,
    'mobilenet0.25': mobilenet0_25,
    'mobilenetv2_1.0': mobilenet_v2_1_0,
    'mobilenetv2_0.75': mobilenet_v2_0_75,
    'mobilenetv2_0.5': mobilenet_v2_0_5,
    'mobilenetv2_0.25': mobilenet_v2_0_25,
    # -----------------------------------------------------------------------------
    # vgg
    # -----------------------------------------------------------------------------
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    # -----------------------------------------------------------------------------
    # resnetv1b_pruned
    # -----------------------------------------------------------------------------
    'resnet18_v1b_0.89': resnet18_v1b_89,
    'resnet50_v1d_0.86': resnet50_v1d_86,
    'resnet50_v1d_0.48': resnet50_v1d_48,
    'resnet50_v1d_0.37': resnet50_v1d_37,
    'resnet50_v1d_0.11': resnet50_v1d_11,
    'resnet101_v1d_0.76': resnet101_v1d_76,
    'resnet101_v1d_0.73': resnet101_v1d_73,
}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % name
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net
