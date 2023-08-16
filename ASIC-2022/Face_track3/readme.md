### 比赛方案：

本赛题是通过构造人脸对抗补丁，实现人脸Black-box adv target attack，本赛题与之前OPPO基本相似，都是通过迁移性的方法构造样本，攻击黑盒模型。与后者不同的是本赛题由于是构造补丁，不限制干扰程度。

`从方法、数据、模型以及训练手段描述`

#### 一、 方法

​	仍然采用传统的黑盒攻击迁移性方法，MI、NI、TI、DI，由于通过观察perturbation的梯度积累，可以发现，在实现target攻击过程中，眼睛眉笔部位积累的梯度更为明显，同时在Dong的2021的文章中，也设计了一种Patch的补丁，防止在人眼的位置。延续使用这样的mask方法（但是这样的方法存在一个弊端，不知道这样的把人眼挡住的patch最后会不会影响到活体检测。

​	创新： 采用多层feature cosine相似度,通过观察不同层之间的相似度影响比较，发现第一层conv和第二个block的变化最为明显，有些偏高，遂对于5层的feature的特征相似度采用loss weight(1,3,1,3,3)。同时考虑不同model的影响程度，设计MSE（mask emsemble model)，将每个model backward的梯度与整体的做相似度比较，取1/cosine作为weight保证current patch不overfit任意一个white-box model。（这个值得考虑一下，因为目前仅做的是value的相似程度，但是最后在update patch的时候选用是sign的方式，后续可以更换一下试验一次）



#### 二、模型

​	目前选用的模型基本机构包括mobile net，facenet, ir50/100/152, ir_se50, iresnet50/100等，初前两个外，都选用的mutil-feature的格式。当前最好分数是在全部一共10个model的共同结果，但是训练效果较慢，可以采用每step random-sampel的随机选取n的model执行当前step的更新。

#### 三、训练策略+参数设置

​	epoch=100，eps = 30, alpha = 30+10/100, batch_size = 1, 



