# Pre-trained models
This repository contains pretrained models. (converted from gluon-cv)

## Environment

- PyTorch 1.1
- Python 3.6
- OpenCV

## Evaluation on imagenet

### resnet

|    Model     | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :----------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| ResNet18_v1  |      70.93      |      89.92      | [70.18](https://drive.google.com/open?id=1kzXeYF4YuetYVANEkYrqhxLJ-7NHsc8E) | 89.52 |
| ResNet34_v1  |      74.37      |      91.87      | [74.04](https://drive.google.com/open?id=13ItQEuuEhtaZo2gM0pQU5pBjAfe3KeW5) | 91.82 |
| ResNet50_v1  |      77.36      |      93.57      | [77.16](https://drive.google.com/open?id=1tAOFeDBG_vreR1TaCEuVHJ9SxZwwYUvV) | 93.56 |
| ResNet101_v1 |      78.34      |      94.01      | [78.23](https://drive.google.com/open?id=1XpsbWY940UaR1klxl83AswzOm1ywCQuc) | 94.09 |
| ResNet152_v1 |      79.22      |      94.64      |                                                              |       |
| ResNet18_v2  |      71.00      |      89.92      | [70.10](https://drive.google.com/open?id=1oS1EFg-ydYGpZUpp_TIDPyN1hYrYY3au) | 89.48 |
| ResNet34_v2  |      74.40      |      92.08      | [74.37](https://drive.google.com/open?id=1Yj1uSTN0CEdUAOIa_sxHUQKEO8OzIhia) | 92.02 |
| ResNet50_v2  |      77.11      |      93.43      | [77.00](https://drive.google.com/open?id=1OyBx5GSYw4xN6Ok4jmyLI9-CEP2BpXDo) | 93.36 |
| ResNet101_v2 |      78.53      |      94.17      | [78.52](https://drive.google.com/open?id=1A68ar0SVU46iVD_tGO5mTPnodnfzWSbD) | 94.15 |
| ResNet152_v2 |      79.21      |      94.31      |                                                              |       |

### resnet_v1b

|      Model      | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :-------------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
|  ResNet18_v1b   |      70.94      |      89.83      | [70.08](https://drive.google.com/open?id=1N8tvBVlMqqfVqQpkNZ31vj4360WKguQj) | 89.44 |
|  ResNet34_v1b   |      74.65      |      92.08      | [74.11](https://drive.google.com/open?id=146cW8hxb6fj161yNeomvjIe5KJl39eAB) | 92.16 |
|  ResNet50_v1b   |      77.67      |      93.82      | [77.57](https://drive.google.com/open?id=1TXEaNlHxgK0BpFFoxeQ9H0cqIYt0yzxL) | 93.58 |
| ResNet50_v1b_gn |      77.36      |      93.59      | [77.22](https://drive.google.com/open?id=1kESi0cdOBR0JmPOhXgaCCnBx99cgKckS) | 93.54 |
|  ResNet101_v1b  |      79.20      |      94.61      | [79.12](https://drive.google.com/open?id=17PVhxH2Frd2yYmg7IAodOt8GPfQzrddJ) | 94.47 |
|  ResNet152_v1b  |      79.69      |      94.74      |                                                              |       |
|  ResNet50_v1c   |      78.03      |      94.09      | [77.89](https://drive.google.com/open?id=1dBnRwuAdkQdKEuF5Vf6ufOY7esrYLF9B) | 94.02 |
|  ResNet101_v1c  |      79.60      |      94.75      |                            79.48                             | 94.72 |
|  ResNet152_v1c  |      80.01      |      94.96      |                                                              |       |
|  ResNet50_v1d   |      79.15      |      94.58      |                                                              |       |
|  ResNet101_v1d  |      80.51      |      95.12      |                                                              |       |
|  ResNet152_v1d  |      80.61      |      95.34      |                                                              |       |

> - `ResNet_v1b` modifies `ResNet_v1` by setting stride at the `3x3` layer for a bottleneck block.
> - `ResNet_v1c` modifies `ResNet_v1b` by replacing the `7x7` conv layer with three `3x3` conv layers.
> - `ResNet_v1d` modifies `ResNet_v1c` by adding an avgpool layer `2x2` with stride `2` downsample feature map on the residual path to preserve more information.

### mobilenet

|      Model       | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :--------------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
|   MobileNet1.0   |      73.28      |      91.30      | [72.85](https://drive.google.com/open?id=1J_mwqonUTvWo0JFM7j2k1SRjPVBCeWT7) | 91.12 |
|  MobileNet0.75   |      70.25      |      89.49      | [69.85](https://drive.google.com/open?id=1T5qQoNJBa9vXnc1e9jo2_Hk4F9kL7qAC) | 89.46 |
|   MobileNet0.5   |      65.20      |      86.34      |                            64.19                             | 85.71 |
|  MobileNet0.25   |      52.91      |      76.94      |                                                              |       |
| MobileNetV2_1.0  |      71.92      |      90.56      |                                                              |       |
| MobileNetV2_0.75 |      69.61      |      88.95      |                                                              |       |
| MobileNetV2_0.5  |      64.49      |      85.47      |                                                              |       |
| MobileNetV2_0.25 |      50.74      |      74.56      |                                                              |       |