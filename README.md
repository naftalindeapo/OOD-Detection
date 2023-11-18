# Out-of-Distribution Detection
This is a simple unified framework for detecting out-of-distribution (OOD) images in neural networks from my [Out-of-Distribution Detection](https://drive.google.com/file/d/1iYIQB629sgECxraShk7qWKXwe9dYhi2e/view?usp=sharing)  research project implemented in [PyThorch](https://pytorch.org). The project explores OOD detection using multiple
techniques, including [MaxSoftmax](https://arxiv.org/abs/1610.02136)), [OpenMax](https://arxiv.org/abs/1511.06233), [Mahalanobis distance](https://arxiv.org/abs/1807.03888), [energy-based methods](https://arxiv.org/abs/2010.03759), and [ODIN](https://arxiv.org/abs/1706.02690), leveraging the pre-trained image classification models [WRN-28-10](https://arxiv.org/abs/1605.07146) and [Dense-BC](https://arxiv.org/abs/1608.06993).

## Experimental Results
To evaluate the performance of the OOD detection methods used in our project a range of metrics, including FPR at 95% TPR, detection error, AUROC, AUPR-In, and AUPR-Out were used. The definition of each metric can be found in the paper. The experimental results are shown as follows.

![alt text](https://drive.google.com/uc?id=1pBQbR1xYrz7bAnBlY8GdzKDMfRoDXUtV)

Below is a detailed visualization of the OOD detection performance results using the five methods across the two models.
![alt text](https://drive.google.com/uc?id=1Rso9pBczr5hr2KT9ANe8JvyyFGyWvtDw) 
![alt text](https://drive.google.com/uc?id=1IkXqoc47m-KrvLoXgOKpo6nhKl4D4I3N) 


## Pre-trained Models
In this project, I used four neural networks: (1.) two DenseNet-BC networks trained on Cifar-10 and Cifar-100 respectively, and (2.) Two Wide ResNet networks trained on Cifar-10 and Cifar-100. The PyTorch implementation of the DenseNet-BC and Wide ResNet are provided by [Andreas Veit](https://github.com/andreasveit/densenet-pytorch), and [Sergey Zagoruyko](https://github.com/szagoruyko/wide-residual-networks), respectively. The in-distribution (ID) test error rates of the two models are given in the table below.
|Architecure     | Cifar-10      | Cifar-100 |
| -------------  |:-------------:| ---------:|
| Dense-BC       | 5.16          | 24.06     |
| WRN-28-10      | 5.93          | 25.10     |
  



