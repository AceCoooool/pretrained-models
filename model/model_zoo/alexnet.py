"""Alexnet, implemented in PyTorch."""

from torch import nn

__all__ = ['AlexNet', 'alexnet']


# -----------------------------------------------------------------------------
# Net
# -----------------------------------------------------------------------------
class AlexNet(nn.Module):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, classes=1000, img_size=224, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.liner = nn.Sequential(
            nn.Linear(256 * (((img_size - 7) // 4 + 1) // 8) ** 2, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True)
        )

        self.output = nn.Linear(4096, classes)

    def forward(self, x):
        x = self.features(x).view(x.shape[0], -1)
        x = self.liner(x)
        x = self.output(x)
        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def alexnet(pretrained=None, **kwargs):
    r"""AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained : str
         the default pretrained weights for model.
    """
    net = AlexNet(**kwargs)
    if pretrained:
        import torch
        net.load_state_dict(torch.load(pretrained))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


if __name__ == '__main__':
    import torch

    a = torch.randn(2, 3, 224, 224)
    net = alexnet()
    print(net)
    with torch.no_grad():
        net(a)
