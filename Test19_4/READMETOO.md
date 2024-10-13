# 注意！！调试时切记要打开的是根目录，不要打开根目录的上级目录，否则相对路径错误


# 注意修改以下内容以使用cuda
```python
# train.py 第95行
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

# trainer.py 第55行
    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

# test.py 第118行
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cpu()
```

# 注意修改以下内容以保证使用多线程
```python
# trainer.py 第37行
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

# test.py 第68行
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
```

# 注意调试时修改以下内容
```
# train.py 
    # 1. 删了required=True，以及加入了default路径，否则调试失败
        parser.add_argument('--cfg',...)
    # 2. 加入default路径 
        parser.add_argument('--output_dir',...)
```

# 其他注意事项
1. --root path只需要定位到data/Synapse
2. configs的yaml文件要加入pretrained_ckpt的路径

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