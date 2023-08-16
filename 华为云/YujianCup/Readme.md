# 数据
目前是讲原数据集放在一起，按照9：1比例分为train:val，同时讲图像沿着width一分为二
# model
尝试了swin，vit ,convnext, efficientnet, 目前最好得分是convnext tiny，input_size为384（也可以换成eff b5 input_size456,dan但是这个训练速度相对较慢，但是flops是在运行范围内的)
也可以继续尝试一下新的model, 一个base的变体（类似于RVT,Wide_resnet这样的）

# 训练
经测试，目前有效的trick是：
1. Mixup（可以100%全用，但是参考一些大的model训练参数，这个mixup 0.5~0.8即可）
2. Label smoothing （单独使用会上分，结合mixup 使用效果不佳，因为mixup这个也有一定的softlabel的作用）
3. loss weight 有用
4. data augmentation: 没啥太大作用，目前最高得分的仅有resize，flip, Normalization，ToTensor
5. R_Drop, 通过KL散度控制同一输出 不同输出（因为有Drop）之间的分布，目测有用
6. 通过对比学习做预训练，效果不佳（可能也有些问题，）
# 推理阶段：
1. 参考训练阶段数据是一分为二的，具体见customize_service.py文件
