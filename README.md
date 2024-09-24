# CODES

## Enviroment
```
python = 3.10
cuda = 11.8
torch = 2.0.1 + cu118
torchvision = 2.0.2 + cu118
numpy
tqdm
tensorboard
tensorboardX
ml-collections
medpy
SimpleITK
scipy
h5py
timm
einops
natten = 0.17.1+torch200cu118 //未使用
```


## 运行
###- Train

```bash
python train.py --dataset Synapse --root_path ./data/Synapse --max_epochs 150 --output_dir ./output  --img_size 224 --base_lr 0.05 --batch_size 24
```

###- Test 
#### 存结果
```bash
python test.py --dataset Synapse --is_saveni --root_path ./data/Synapse --max_epoch 150 --output_dir ./output --img_size 224 --base_lr 0.05 --batch_size 24
```

#### 不存结果
```bash
python test.py --dataset Synapse --root_path ./data/Synapse --max_epoch 150 --output_dir ./output --img_size 224 --base_lr 0.05 --batch_size 24
```
