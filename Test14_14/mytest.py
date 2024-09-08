"""
    自用测试文件
"""


def showImage1():
    """
        显示npz文件图片和标签
    """
    import numpy as np
    import matplotlib.pyplot as plt

    path = "data\Synapse\\train_npz\case0005_slice030.npz"
    data = np.load(path)
    img_data1 = data['image']
    seg_data1 = data['label']

    b,s ,t,w= np.unique(seg_data1,return_counts=True,return_index=True,return_inverse=True)
    print(b) # 元素
    print(s) # 第一个的index
    print(t) # 逆转换索引
    print(w) # 元素个数

    plt.figure(figsize=(10,5)) #设置窗口大小
    plt.subplot(1,2,1)
    plt.title('image')
    plt.imshow(img_data1)

    plt.subplot(1,2,2)
    plt.title('label')
    plt.imshow(seg_data1)
    plt.show()

def showTestImg1():
    """
        显示xxx.npy.h5文件的每张切片
    """
    import h5py
    import matplotlib.pyplot as plt
    with h5py.File('data/Synapse/test_vol_h5/case0001.npy.h5', 'r') as f:
        img = f['image']
        lab = f['label']
        # print(img) # 显示img的规格

        slice_index = 147  # 可以修改这个索引值来查看不同的切片
        slice_img = img[:, :, slice_index]
        slice_lab = lab[:, :, slice_index]
        
        # 显示切片
        plt.figure(figsize=(10,10))
        plt.subplot(2,1,1)
        plt.imshow(slice_img, cmap='gray')
        # plt.colorbar()
        plt.title(f'Slice at index {slice_index}')

        plt.subplot(2,1,2)
        plt.imshow(slice_lab, cmap='gray')
        # plt.colorbar()
        plt.title(f'Slice at index {slice_index}')
        plt.show()

def showPredImg():
    """
        查看xxx.nii.gz文件，查看图片，标签以及预测
    """
    import nibabel as nib
    import matplotlib.pyplot as plt

    # 加载.nii.gz文件
    i = '0003'
    img = nib.load(f'predictions\predictions\case{i}_img.nii.gz')
    gt = nib.load(f'predictions\predictions\case{i}_gt.nii.gz')
    pred = nib.load(f'predictions\predictions\case{i}_pred.nii.gz')

    # 提取图像数据
    img_data = img.get_fdata()
    gt_data = gt.get_fdata()
    pred_data = pred.get_fdata()

    # 显示图像的一些基本信息
    print(f"图像数据形状：{img_data.shape}")
    print(f"像素数据类型：{img_data.dtype}")

    # 可视化图像切片
    # slice_index = img_data.shape[-1] // 2  # 选择中间切片
    slice_index = 100
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.imshow(img_data[:, :, slice_index],cmap='gray') # 参数cmap='gray'设为灰度图
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(gt_data[:, :, slice_index])
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_data[:, :, slice_index])
    plt.axis('off')
    plt.show()

def vis_save(original_img, pred, save_path):
    import numpy as np
    import cv2
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    pred = pred.astype(np.uint8)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    original_img = np.where(pred==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, red   ), original_img)
    original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(pred==6, np.full_like(original_img, yellow), original_img)
    original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)

def predAndImg():
    """
        将预测图与原图叠加
    """
    import nibabel as nib

    # 加载.nii.gz文件
    i = '0003'
    img = nib.load(f'predictions\predictions\case{i}_img.nii.gz')
    gt = nib.load(f'predictions\predictions\case{i}_gt.nii.gz')
    pred = nib.load(f'predictions\predictions\case{i}_pred.nii.gz')

    img_data = img.get_fdata()
    gt_data = gt.get_fdata()
    pred_data = pred.get_fdata()

    slice_index = 100
    img_slice = img_data[:, :, slice_index]
    pred_slice = pred_data[:, :, slice_index]
    print(pred_slice)

    vis_save(img_slice, pred_slice, 'cv2.jpg')


def test():
    from networks.NewModel import SwinTransformerSys as net
    import torch

    x = torch.randn(1,3,224,224)
    model = net()
    out = model(x)
    print(1)

test()