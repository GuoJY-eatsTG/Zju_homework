# Zju_homework
The model used for this homework is DenseNet.
## Dataset
Cifar 10:Downloadable from http://www.cs.toronto.edu/~kriz/cifar.html
## Requirements
* Python = 3.7
* torch ~= 1.6.0
* torchvision ~= 0.6.0
## Useage
```sh
#learning rate default = 0.1
python main.py --lr 0.1
#weight_decay default = 5e-4
python main.py --weight_decay=5e-4
#momentum default = 0.9
python main.py --momentum=0.9
```

## Results  
|      parameters |                    |                          |                    |                    |                          |
|:---------------:|:------------------:|:------------------------:|:------------------:|:------------------:|:------------------------:|
|        lr       |         0.1        |          0.3             |        0.5         |         0.7        |          0.9             |
|      test_ACC   |        95.760%     |           93.340%        |        90.800%     |        85.700%     |           80.720%        |

  
**Table 1: DenseNet Performance comparison of different lr on the dataset cifar10(weight_decay=5e-4,momentum=0.9).**

|      parameters |                    |                          |                    |
|:---------------:|:------------------:|:------------------------:|:------------------:|
|  weight_decay   |         5e-5       |          5e-4            |        5e-3        |
|   test_ACC      |        94.720%     |           95.760%        |        79.350%     |

  
**Table 2: DenseNet Performance comparison of different weight_decay on the dataset cifar10(lr=0.1,momentum=0.9).**


|      parameters |                    |                          |                    |
|:---------------:|:------------------:|:------------------------:|:------------------:|
|    momentum     |         0.5        |          0.7             |        0.9         |
|      test_ACC   |        95.270%     |           95.580%        |        95.760%     |

  
**Table 3: DenseNet Performance comparison of different momentum on the dataset cifar10(weight_decay=5e-4,lr=0.1).**
