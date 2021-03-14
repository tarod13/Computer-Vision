# Computer Vision implementations and experiments in PyTorch

## Papers reviewed so far
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015.
* [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](http://arxiv.org/abs/1805.01978) by Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin, 2018.

## Accuracies obtanied (Top-1)
| Model                                          | Type                | CIFAR10   | CIFAR100  |
| ---------------------------------------------- | ------------------- | --------- | --------- |
| Vanilla CNN                                    | Supervised          | 60.90%    | 37.98%    |
| ResNet-18                                      | Supervised          | 90.10%    | 58.65%    |
| ResNet-32                                      | Supervised          | 90.89%    | --        |
| Instance Disc. Parametric Classifier           | Unsupervised + KNN  | 60.28%    | --        |
| Instance Disc. Non-parametric Classifier       | Unsupervised + KNN  | 77.46%    | --        |
| Instance Disc. Non-parametric Classifier (NCE) | Unsupervised + KNN  | 72.12%    | --        |
