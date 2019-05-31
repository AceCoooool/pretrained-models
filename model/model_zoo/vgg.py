"""VGG, implemented in PyTorch."""
# TODO: add weight init
from __future__ import division

__all__ = ['VGG', 'get_vgg',
           'vgg11', 'vgg13', 'vgg16', 'vgg19',
           'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
           ]

from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class VGG(nn.Module):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should be one larger than layers.
    classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters) - 1
        self.features = self._make_features(layers, filters, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(filters[-1] * 7 ** 2, 4096),
            nn.ReLU(inplace=True), nn.Dropout(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=True),
            nn.Linear(4096, classes),
        )

    @staticmethod
    def _make_features(layers, filters, batch_norm):
        featurizer = list()
        for i, num in enumerate(layers):
            for j in range(num):
                featurizer.append(nn.Conv2d(filters[i] if j == 0 else filters[i + 1],
                                            filters[i + 1], kernel_size=3, padding=1))
                if batch_norm:
                    featurizer.append(nn.BatchNorm2d(filters[i + 1]))
                featurizer.append(nn.ReLU(inplace=True))
            featurizer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*featurizer)

    def forward(self, x):
        b = x.shape[0]
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 7).view(b, -1)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# Specification
# -----------------------------------------------------------------------------
vgg_spec = {11: ([1, 1, 2, 2, 2], [3, 64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [3, 64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [3, 64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [3, 64, 128, 256, 512, 512])}


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def get_vgg(num_layers, pretrained=None, **kwargs):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    pretrained : str
        the default pretrained weights for model.
    """
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    if pretrained:
        import torch
        net.load_state_dict(torch.load(pretrained))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def vgg11(**kwargs):
    r"""VGG-11 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    return get_vgg(11, **kwargs)


def vgg13(**kwargs):
    r"""VGG-13 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    return get_vgg(13, **kwargs)


def vgg16(**kwargs):
    r"""VGG-16 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_vgg(16, **kwargs)


def vgg19(**kwargs):
    r"""VGG-19 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    return get_vgg(19, **kwargs)


def vgg11_bn(**kwargs):
    r"""VGG-11 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    kwargs['batch_norm'] = True
    return get_vgg(11, **kwargs)


def vgg13_bn(**kwargs):
    r"""VGG-13 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    kwargs['batch_norm'] = True
    return get_vgg(13, **kwargs)


def vgg16_bn(**kwargs):
    r"""VGG-16 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    kwargs['batch_norm'] = True
    return get_vgg(16, **kwargs)


def vgg19_bn(**kwargs):
    r"""VGG-19 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    pretrained : str
        the default pretrained weights for model.
    """
    kwargs['batch_norm'] = True
    return get_vgg(19, **kwargs)


if __name__ == '__main__':
    import torch

    a = torch.randn(2, 3, 224, 224)

    net1 = vgg11()
    # net2 = vgg11_bn()
    # net3 = vgg13()
    # net4 = vgg13_bn()
    # net5 = vgg16()
    # net6 = vgg16_bn()
    # net7 = vgg19()
    # net8 = vgg19_bn()

    with torch.no_grad():
        net1(a)
        # net2(a)
        # net3(a)
        # net4(a)
        # net5(a)
        # net6(a)
        # net7(a)
        # net8(a)
