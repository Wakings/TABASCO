# Label-Noise Learning with Intrinsically Long-Tailed Data

**Abstract:** Label noise is one of the key factors that lead to the poor generalization of deep learning models. Existing label-noise learning methods usually assume that the ground-truth classes of the training data are balanced. However, the real-world data is often imbalanced, leading to the inconsistency between observed and intrinsic class distribution with label noises. In this case, it is hard to distinguish clean samples from noisy samples on the intrinsic tail classes with the unknown intrinsic class distribution. In this paper, we propose a learning framework for label-noise learning with intrinsically long-tailed data. Specifically, we propose two-stage bi-dimensional sample selection (TABASCO) to better separate clean samples from noisy samples, especially for the tail classes. TABASCO consists of two new separation metrics that complement each other to compensate for the limitation of using a single metric in sample separation. Extensive experiments on benchmarks we proposed with real-world noise and intrinsically long-tailed distribution demonstrate the effectiveness of our method. 

### Update！！！

- release the codes to construct the datasets
- release the accelerated version of TABASCO

### Usage

Here is an example shell script to run TABASCO on CIFAR-10 :

```python
#!/bin/bash
dataset='cifar10'
num_class=10
imb_factor='0.1'
noise_mode='unif'
corruption_prob='0.4'
python3 ./experiment/train_cifar_ssl.py \
			--dataset ${dataset}  \
			--num_class ${num_class}  \
			--imb_factor ${imb_factor}   \
			--corruption_type ${noise_mode}  \
			--corruption_prob ${corruption_prob}  
```



