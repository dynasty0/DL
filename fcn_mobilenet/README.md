# fcn: using mobilenet instead of vgg16

This implementation is refered to [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)

But we using [mobilenet](https://arxiv.org/abs/1704.04861) to extract features instead of vgg16.

It is much faster than the traditional [fcn](https://github.com/dynasty0/DL/tree/master/fcn)

## Usage

### Train
```
python train.py
```

### Test
```
python test.py xxx.jpg
```

## Example

### Input image1

![Input1](https://github.com/dynasty0/DL/blob/master/fcn_mobilenet/imgs/input1.jpg)

### Output image1

![Output1](https://github.com/dynasty0/DL/blob/master/fcn_mobilenet/imgs/output1.png)

### Input image2

![Input2](https://github.com/dynasty0/DL/blob/master/fcn_mobilenet/imgs/input2.jpg)

### Output image2

![Output2](https://github.com/dynasty0/DL/blob/master/fcn_mobilenet/imgs/output2.png)