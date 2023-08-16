## 代码说明

### 环境配置

Python 版本：3.9.12 PyTorch 版本：1.11.0 CUDA 版本：11.3

所需环境在 `requirements.txt` 中定义。

### 数据

- 仅使用大赛提供的有标注数据（10万）和 大赛提供的无标注数据（预训练）。
- 未使用任何额外数据。

### 预训练模型

- 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

### 算法描述

- **数据加载：**使用baseline中的data_helper.py中MultiModalDataset加载数据。对于文本数据，分布使用BertTokenizer(加载上述预训练模型)对标题，ocr，acr分别获取input_ids，其中每类长度限制为100。视频数据加载过程未做修改。
- **模型结构：**参考2021年[QQ浏览器2021 AI 算法大赛，赛道1](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st) 中双流model结构做了部分修改。首先将文本分别送入个Embedding层，视频特征通过一个全连接层与senet层再输入另一个embedding，让将video_embeddings和text_embeddings通过cat连接起来，与cat之后的masks一同输入到BertEncoder，Encoder输出结果通过MLP进行分类。

### 训练流程

根目录下执行sh train.sh（目前目录中无预训练模型权重，请组委会联系我们提供预训练模型，否则train部分的代码会由于预训练模型的缺失而报错）

- **预训练：**首先需要对模型进行预训练，直接运行src目录下的`pretrain_model.py`即可开始预训练。预训练采用了mlm,mfm,itm三个任务。模型在大赛提供的无标注数据集上做预训练。预训练过程中每5000个step保存一次模型，我们最终使用的是第100000个step时的模型。

  (1) 图像文本匹配任务(itm)

  将视频特征后一半的特征逆序，文本标志位对应输出使用一个线性的ITM head将输出feature映射成一个二值logits，用来判断视频特征和文本是否匹配。然后计算BCE loss作为ITM的损失函数(交叉熵)。

  (2) Mask language model 任务（mlm）
  与常见的自然语言处理 mlm 预训练方法相同，对 text 随机 15% 进行 mask，预测 mask 词。
  多模态场景下，结合视频的信息预测 mask 词，可以有效融合多模态信息。

  (3) Mask frame model 任务（mfm）
  对 frame 的随机 15% 进行 mask，mask 采用了全 0 的向量填充。
  考虑到 frame 为连续的向量，难以类似于 mlm 做分类任务。
  借鉴了对比学习思路，希望 mask 的预测帧在整个 batch 内的所有帧范围内与被 mask 的帧尽可能相似。
  采用了 Nce loss，最大化 mask 帧和预测帧的互信息

  预训练的损失函数是如上三个任务各自损失函数的加权和，具体参考`model_pretrain.py`。

- **训练**：对有标注的数据集进行K折训练（K=10），每折加载预训练权重然后进一步训练。具体见`main_pretrain.py`代码

### 推理流程

执行sh inference.sh

- 选择10折训练后在验证集上效果最好的model推理测试数据，并在相应的目录中保存中间推理结果（logits），然后通过模型融合，将10折logits相加，再取argmax作为最后结果，生成result.csv。

