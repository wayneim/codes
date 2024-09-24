# codes
# 运行
- Train

```bash
python train.py --dataset Synapse --root_path ./data/Synapse --max_epochs 150 --output_dir ./output  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
python test.py --dataset Synapse --is_saveni --root_path ./data/Synapse --max_epoch 150 --output_dir ./output --img_size 224 --base_lr 0.05 --batch_size 24
```

python test.py --dataset Synapse --root_path ./data/Synapse --max_epoch 150 --output_dir ./output --img_size 224 --base_lr 0.05 --batch_size 24
