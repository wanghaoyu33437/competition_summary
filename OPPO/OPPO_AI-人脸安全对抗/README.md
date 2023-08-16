#  详细说明

### 1. 环境配置

| 操作系统 | Ubuntu 20.04 lts |
| -------- | ---------------- |
| Python   | 3.7.10           |
| CUDA     | 11.2             |
| PyTorch  | 1.8.0            |

其他所依赖第三方库及版本见requirements.txt。

因torch版本问题，在执行完pip install -r requirements.txt后，需执行下面语句：

```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. 模型相关

​	我们采用集成模型的方式训练人脸对抗样本，模型结构包括FaceNet, MobileFaceNet, IR_101,IR_152, IR_50, IRSE_50等，IRSE_50是在IR_50的基础上了SE(Squeeze-and-Excitation)，可以理解为加了一个Attention。

​	因为是黑盒攻击，所以采用Backbone提取人脸特征，用于分类。

| 模型名称      | 简介                                                | 参考链接（code）                                             |
| :------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| Facenet       | Inception Resnet V1作为Backbone                     | https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py |
| MobileFaceNet | mobilenet替换了facenet的inception，进行提取人脸特征 | https://github.com/JDAI-CV/FaceX-Zoo/blob/main/backbone/MobileFaceNets.py |
| IR_101        | ResNet101作为BackBone                               | https://gitee.com/heronwang/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py |
| IR_152        | ResNet152作为BackBone                               | https://gitee.com/heronwang/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py |
| IR_50         | ResNet50作为BackBone                                | https://gitee.com/heronwang/face.evoLVe.PyTorch/blob/master/backbone/model_irse.py |

网络模型定义结构位于models目录下facenet.py，irse.py等文件。针对不同的model，我们选择加载在不同训练集上训练的权重，**facenet，mobileFaceNet，irse50，IR_152**加载在LFW上训练参数（https://drive.google.com/drive/folders/1G_2R_7XQhzzMQdEhph0ZI7dV4sGYjjzu），**ir50_ms1m，IR_50_LFW_AdvTrain，IR_101**，分别加载baseline中model参数。具体加载方式可见run.py中set_model_info方法。

我们已为所需加载的model_list设置了默认参数，每次运行时会首先加载上述7个model。

### 3 . 训练相关

 	本次比赛中我们团队没有采用自行训练的model，所以在训练相关中不做描述。

### 4. 解决方案和算法介绍

 	针对本次比赛赛题，由于每日上传次数的限制，我们从基于迁移性的人脸对抗攻击出发，以FGSM为基本方法，结合目前提高攻击迁移性方法 **momentum(MI)**，**input-diversity**（**DI**），**ensemble-model**，**translation-Invariant**（**TI**），**Patch-wise(PI)**， **Staircase Sign** **Method(SSM)**等（这些方法将在后面具体介绍，包括公式，paper，作者等）。

​	算法流程：

1. 输入输出阶段

   ​	我们通过将攻击图像adv_image和face pair的比对图像com_image（题目中提及到的样本*I*构建的face pair的比对）输入model，在输入阶段我们参考了**【CVPR2019】Improving transferability of adversarial examples with input diversity**（Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, Alan L. Yuille)，对adv_image和com_image做同样的resize和padding后使用interpolate方法将图像调整为model输入尺寸，然后输入model，最后输出[1,512]的特征值。

2. LOSS及权重

   ​	使用**余弦相似度**计算adv_image的特征与com_image的特征距离作为攻击强度目标（攻击损失-adv_loss），另一方面使用**MSSSIM**计算adv_image与ori_image的结构相似度（质量损失-struct_loss）将这两部分作为total loss用于更新adv_image。

   我们将迭代次数设置为40次，在每次迭代后根据当前的adv_loss和struct_loss更新不同损失权重： 

   ​	![1634976476539](C:\Users\WHY\AppData\Roaming\Typora\typora-user-images\1634976476539.png)

3. 梯度计算

   ​	根据loss可以得到gradient，第一步使用了动量的方法，参考了**【CVPR2018】Boosting adversarial attacks with momentum**（Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, Jianguo Li）我们将此次计算得到的gradient与上次迭代的gradient结合，

   ![1634978284239](C:\Users\WHY\AppData\Roaming\Typora\typora-user-images\1634978284239.png)

   然后第二步参考了**【CVPR2019】Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks**(Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu)，将梯度做一次平移转换（考虑CNN的平移不变性）

   ![1634978301618](C:\Users\WHY\AppData\Roaming\Typora\typora-user-images\1634978301618.png)

   第三步参考**【ECCV2020】Patch-wise Attack for Fooling Deep Neural Network**（Lianli Gao，Qilong Zhang，Jingkuan Song，Xianglong Liu，Heng Tao Shen），这个方法是将当前区域干扰超出设置值的部分映射到周围区域（减少直接clip掉的数值造成的信息丢失），同时这部分参考**Staircase Sign Method for Boosting Adversarial Attacks**（Lianli Gao, Qilong Zhang, Xiaosu Zhu， Jingkuan Song Heng Tao Shen）将sign值做一个百分值计算，这一步的作用是不仅仅考虑gradient的方向，同时考虑数值的影响。（计算参数直接参考文章中参数设置）

   ![1634978720202](C:\Users\WHY\AppData\Roaming\Typora\typora-user-images\1634978720202.png)

4. adv_image选取

   我们最开始设置的干扰大小的是8，MSSSIM是0.96。迭代40次得到的adv_image，我们使用Face++的人脸比对API验证当前adv_image与com_image的相似程度，当返回值大于65时，将干扰值+1（最大18），MSSSIM-0.01（最小0.88）重新训练，直到返回值小于65.

**代码运行时需要注意点：**

1. 由于内存或者其他问题造成运行中断时，需删除当前未训练完成image
2. 处理小图像时大概需要10G显存，大图像14G左右。
3. 可以在run.sh中修改使用--cuda 选择显卡，因为代码中设置了对于已生成图像直接continue，所以可多次运行脚本。
4. 因为result_data中已经生成了1000张对图像，重新生成样本时需要将原有图像删除或者直接更改result_data/images为result_data/images1即可





