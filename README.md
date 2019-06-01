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
|  ResNet152_v1b  |      79.69      |      94.74      |                            78.07                             | 93.97 |
|  ResNet50_v1c   |      78.03      |      94.09      | [77.89](https://drive.google.com/open?id=1dBnRwuAdkQdKEuF5Vf6ufOY7esrYLF9B) | 94.02 |
|  ResNet101_v1c  |      79.60      |      94.75      | [79.48](https://drive.google.com/open?id=1JBc1TmOf95rubOWu8hOz_OXBIOtUV6L4) | 94.72 |
|  ResNet152_v1c  |      80.01      |      94.96      |                            78.18                             | 93.99 |
|  ResNet50_v1d   |      79.15      |      94.58      | [79.04](https://drive.google.com/open?id=1oMrJ3U45ERi1EOCHTc5cjOba9hj-v4Os) | 94.61 |
|  ResNet101_v1d  |      80.51      |      95.12      | [80.52](https://drive.google.com/open?id=1pWuT_iipgk6I_dM1NWAuxh93VQzz9HaA) | 95.23 |
|  ResNet152_v1d  |      80.61      |      95.34      | [80.75](https://drive.google.com/open?id=1mElrSlUvCR3bpnc6GHYUC2via3-THWSq) | 95.34 |

> - `ResNet_v1b` modifies `ResNet_v1` by setting stride at the `3x3` layer for a bottleneck block.
> - `ResNet_v1c` modifies `ResNet_v1b` by replacing the `7x7` conv layer with three `3x3` conv layers.
> - `ResNet_v1d` modifies `ResNet_v1c` by adding an avgpool layer `2x2` with stride `2` downsample feature map on the residual path to preserve more information.

### mobilenet

|      Model       | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :--------------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
|   MobileNet1.0   |      73.28      |      91.30      | [72.85](https://drive.google.com/open?id=1J_mwqonUTvWo0JFM7j2k1SRjPVBCeWT7) | 91.12 |
|  MobileNet0.75   |      70.25      |      89.49      | [69.85](https://drive.google.com/open?id=1T5qQoNJBa9vXnc1e9jo2_Hk4F9kL7qAC) | 89.46 |
|   MobileNet0.5   |      65.20      |      86.34      | [64.19](https://drive.google.com/open?id=1cUBh3kfq0hAi6FuATYE5axP_oK9oC8VQ) | 85.71 |
|  MobileNet0.25   |      52.91      |      76.94      | [51.09](https://drive.google.com/open?id=1rGcC_6ehRuBkeMwODIhCnRmI1WlbuffU) | 75.36 |
| MobileNetV2_1.0  |      71.92      |      90.56      | [71.78](https://drive.google.com/open?id=184i133xDNAKQ03hSwUwAFeZIavrft0kF) | 90.36 |
| MobileNetV2_0.75 |      69.61      |      88.95      | [69.29](https://drive.google.com/open?id=1Yj6cIOUExRiKGeA4-Ky6linzI06R11GA) | 88.81 |
| MobileNetV2_0.5  |      64.49      |      85.47      | [64.15](https://drive.google.com/open?id=1Io_tsEmwz7yF41UPpgVRcYLJMyV4Vyhw) | 85.40 |
| MobileNetV2_0.25 |      50.74      |      74.56      | [50.14](https://drive.google.com/open?id=1-q81iQvR6UROcDFipOZqATEXSv64qOYN) | 74.13 |

### vgg

|  Model   | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
|  VGG11   |      66.62      |      87.34      | [67.26](https://drive.google.com/open?id=12NuWE6hmnAu2FVZWTLhRKeqDSqUEsbun) | 87.73 |
|  VGG13   |      67.74      |      88.11      | [68.15](https://drive.google.com/open?id=16xTQJB1RdCOTEdNA9rfrb-l-EaBp4F8n) | 88.47 |
|  VGG16   |      73.23      |      91.31      | [70.09](https://drive.google.com/open?id=1Qojl0JgORqlrzJ-fH3BfYGw1fAaum-Va) | 89.70 |
|  VGG19   |      74.11      |      91.35      | [70.86](https://drive.google.com/open?id=1yLN2RHTEgg0YoYink2GQwVqMgZYvH8KC) | 90.17 |
| VGG11_bn |      68.59      |      88.72      | [68.94](https://drive.google.com/open?id=1Vwhp6e19wkoywpb3U0KJL2aHtBVIgGZb) | 88.88 |
| VGG13_bn |      68.84      |      88.82      | [69.51](https://drive.google.com/open?id=1WnFNR4diCCzG3zy2_GdKcPsl8w_cDxjF) | 89.46 |
| VGG16_bn |      73.10      |      91.76      | [72.07](https://drive.google.com/open?id=1-2qaUQXVChIyQ8GoLkPt_CeMMa7CvS4t) | 90.97 |
| VGG19_bn |      74.33      |      91.85      | [72.85](https://drive.google.com/open?id=1zHpPha3jkmulUetEA8YbkxcoljVnaPwq) | 91.26 |

> Note: the vgg model here is converted from torchvision

### resnext

|        Model        | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :-----------------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
|   ResNext50_32x4d   |      79.32      |      94.53      | [79.41](https://drive.google.com/open?id=1cjysurZtflI6emTfQUCT3x8JHISAGUlX) | 94.54 |
|  ResNext101_32x4d   |      80.37      |      95.06      | [80.52](https://drive.google.com/open?id=1E6W0XGAzDPs9zzV-AtjdOw-FoHuDFO1E) | 95.20 |
|  ResNext101_64x4d   |      80.69      |      95.17      | [80.84](https://drive.google.com/open?id=1ygaTFO75UYM8eaWJ-Y1MQ6OHfHoXgfwo) | 95.27 |
| SE_ResNext50_32x4d  |      79.95      |      94.93      | [80.17](https://drive.google.com/open?id=1qFwRuFvcmRvmUqdjyBdUcNLVnxla3QDU) | 94.97 |
| SE_ResNext101_32x4d |      80.91      |      95.39      | [81.27](https://drive.google.com/open?id=16TOK78CZrKFjCCiZXXSJ6zcrrMblnkg7) | 95.42 |
| SE_ResNext101_64x4d |      81.01      |      95.32      | [81.19](https://drive.google.com/open?id=1cc21-njLLCJrAt-osUfJXKch12PHnPlI) | 95.60 |

### resnetv1b_pruned

|       Model        | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :----------------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| resnet18_v1b_0.89  |      67.2       |      87.45      | [65.78](https://drive.google.com/open?id=1nK09yCWXg2Q-N6qot86Rrx21cUndwEZu) | 86.63 |
| resnet50_v1d_0.86  |      78.02      |      93.82      | [77.61](https://drive.google.com/open?id=1m5WQEW2sjegQZm7UJn9H1T5Hl5vG-ye0) | 93.90 |
| resnet50_v1d_0.48  |      74.66      |      92.34      | [74.10](https://drive.google.com/open?id=1F4-dLyHDjcw3eID9BTStrMtMfYkn_MjJ) | 92.10 |
| resnet50_v1d_0.37  |      70.71      |      89.74      | [69.47](https://drive.google.com/open?id=1AJ2lN4dqCNWOZw6grc3wyZz75-VGeVbm) | 89.12 |
| resnet50_v1d_0.11  |      63.22      |      84.79      | [61.12](https://drive.google.com/open?id=1kLb4p3UB0Ern9OxfCrAKEeMLX2YnD-0t) | 83.31 |
| resnet101_v1d_0.76 |      79.46      |      94.69      | [79.55](https://drive.google.com/open?id=1YR88eeBw8QMTP0J17u8xBT1aaiAKPnFh) | 94.81 |
| resnet101_v1d_0.73 |      78.89      |      94.48      | [78.68](https://drive.google.com/open?id=19aXUGH9nneXP62UbCHTtaIRNmnhwv6tN) | 94.41 |

### squeezenet

|     Model     | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :-----------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| SqueezeNet1.0 |      56.11      |      79.09      | [55.67](https://drive.google.com/open?id=1Ux-VwK6Sa33gKtzi0BhQdPduwPcTEJ8I) | 78.47 |
| SqueezeNet1.1 |      54.96      |      78.17      | [55.27](https://drive.google.com/open?id=1UFa1Z2G0LWNwYZu_M-r_aGaUpBzWiJht) | 78.55 |

### densenet

|    Model    | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :---------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| DenseNet121 |      74.97      |      92.25      | [74.65](https://drive.google.com/open?id=1B8I0s9HYUhg4IqpBJJucqOGeafKRWJJS) | 92.15 |
| DenseNet161 |      77.70      |      93.80      | [77.64](https://drive.google.com/open?id=1PzWbaaYi_TWIFGWOrNAtfdE8s-V3uMew) | 93.97 |
| DenseNet169 |      76.17      |      93.17      | [76.26](https://drive.google.com/open?id=1oFiS0WZTImshI8ALUGPA0lSjZyD6WP5S) | 93.18 |
| DenseNet201 |      77.32      |      93.62      | [77.64](https://drive.google.com/open?id=1A_5Fg4yzo8UH9qyzmpCsocd6bwxWlNEW) | 93.97 |

### inception

|    Model    | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :---------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| InceptionV3 |      78.77      |      94.39      | [78.62](https://drive.google.com/open?id=1t3YhPYr571OsmdbAF4huSMDxu_UOIUZ6) | 94.42 |

> `InceptionV3` is evaluated with input size of 299x299.

### alexnet

|  Model  | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :-----: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| AlexNet |      54.92      |      78.03      | [54.28](https://drive.google.com/open?id=1eAVM1Ic2ytAR40aOhhhDuNUIhtzIW5a9) | 77.68 |

### darknet

|   Model   | Acc@1(gluon-cv) | Acc@5(gluon-cv) |                            Acc@1                             | Acc@5 |
| :-------: | :-------------: | :-------------: | :----------------------------------------------------------: | :---: |
| darknet53 |      78.56      |      94.43      | [78.54](https://drive.google.com/open?id=1b5KHVz1FY8MHyHTlRZU5OdgL5t3jHYdT) | 94.54 |



## TODO

- [ ] Add more pretrained models