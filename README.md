# Env
```
Python==3.8.10
torch==1.11.0
```

# MNIST
```
python main.py --balanced --batch-size 500 --test-batch-size 500 --sigma 0.0001 \
               --K 20 --M 1000 --dataset mnist --arch mnist_model \
               --pretrained mnist_ckpts/best.ckpt --check --measure-speed
```

# CIFAR-10

### ELLA
```
python main.py --balanced --batch-size 500 --test-batch-size 500  \
               --sigma2 0.04 --K 20 --M 2000 --search-freq 1000 \
               --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```

### LLA*
```
python laplace_baseline.py --batch-size 200 --test-batch-size 200 \
                           --subset-of-weights last_layer --hessian-structure full --job-id lastl-full \
                           --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```

### LLA*-KFAC
```
python laplace_baseline.py --batch-size 200 --test-batch-size 200 \
                           --subset-of-weights last_layer --hessian-structure kron --job-id lastl-kron \
                           --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```

### LLA-Diag
```
python laplace_baseline.py --batch-size 200 --test-batch-size 200 \
                           --subset-of-weights all --hessian-structure diag --job-id all-diag \
                           --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```

### LLA-KFAC
```
python laplace_baseline.py --batch-size 200 --test-batch-size 200 \
                           --subset-of-weights all --hessian-structure kron --job-id all-kron \
                           --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```

### MFVI-BF
```
python mfvi_baseline.py --batch-size 256 --test-batch-size 256 --dataset cifar10 \
                        --epochs 12 --lr 1e-3 --ft_lr 1e-4 --decay 0.0005 \
                        --arch [cifar10_resnet20, cifar10_resnet32, cifar10_resnet44, cifar10_resnet56]
```


# ImageNet

### ELLA
```
python main.py --balanced --batch-size 100 --test-batch-size 100 \
               --sigma2 0.01 --K 20 --M 2000 --I 100 --search-freq 100 --dataset imagenet \
               --arch [resnet18, resnet34, resnet50]
```

### MFVI-BF
```
python mfvi_baseline.py --batch-size 128 --test-batch-size 256 --dataset imagenet \
                        --epochs 4 --lr 1e-3 --ft_lr 1e-4 --decay 0.0001 \
                        --arch [resnet18, resnet34, resnet50]
```

### ELLA on ViT-B
```
python main.py --balanced --batch-size 100 --test-batch-size 100 \
               --sigma2 0.00001 --K 20 --M 2000 --I 80 --search-freq 100 --dataset imagenet \
                --arch vit_base_patch16_224
```
