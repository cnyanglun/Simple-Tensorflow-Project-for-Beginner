# 服装图像分类 · Fashion-MNIST

[教程王者](https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn)

用 TensorFlow / Keras 训练一个神经网络，把 28×28 的服装灰度图分成 10 类。

这是一个**教学项目**：代码量很小，但覆盖了图像分类的完整闭环 —— 数据 → 预处理 → 建模 → 训练 → 评估 → 预测 → 可视化。即使不理解所有细节也没关系，每一步都有对应的 Notebook 小节和输出可以对照。

数据集是 [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)：6 万张训练图 + 1 万张测试图，10 个类别。

| 标签 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:---|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 类别 | T恤/上衣 | 裤子 | 套头衫 | 连衣裙 | 外套 | 凉鞋 | 衬衫 | 运动鞋 | 包 | 短靴 |

## 项目结构

```
clothing-image-classifier/
├── clothes-classifier.ipynb   # 主线：完整流程 + 讲解 + 可视化（从这里开始看）
├── app.py                     # 精简版脚本：只做「加载数据 → 预处理 → 看第一张图」
├── readme.md                  # 本文档
└── .vscode/settings.json      # 指定用 conda 环境运行
```

| 文件 | 作用 |
|:---|:---|
| `clothes-classifier.ipynb` | 教学主线，按小节推进，每个 cell 都有输出可对照 |
| `app.py` | 把 Notebook 前半部分抽成脚本，用来快速确认环境和数据是否就绪（不含训练） |

## 极简流程

```
① 加载数据 → ② 预处理 → ③ 构建模型 → ④ 编译 → ⑤ 训练 → ⑥ 评估 → ⑦ 预测 → ⑧ 可视化
```

| # | 步骤 | Notebook 小节 | 做什么 | 关键点 |
|:--|:---|:---|:---|:---|
| 1 | 环境检查 | 开头两个 cell | 导入 TF、打印版本、列出可用设备 | `tf.config.list_physical_devices('GPU')` |
| 2 | 加载数据 | `## 导入Fashion MNIST 数据集` | `fashion_mnist.load_data()` 下载并解包 | 训练集 `(60000, 28, 28)`，测试集 1 万张 |
| 3 | 浏览数据 | `## 浏览数据` | 看 `.shape`、`len()`、`np.unique()` | 确认形状、数量、标签范围 0~9 |
| 4 | 预处理 | `## 数据预处理` | `train_images / 255.0` | 像素从 0~255 缩到 0~1，训练更稳 |
| 5 | 构建模型 | `## 构建模型` | `Sequential([Flatten, Dense(128), Dense(10)])` | 展平 → 隐藏层 → 10 类输出 |
| 6 | 编译 | `## 编译模型` | `adam` + `SparseCategoricalCrossentropy(from_logits=True)` | 损失函数要配合"输出 logits"的写法 |
| 7 | 训练 | 编译模型下一格 | `model.fit(..., epochs=10)` | 训练准确率 0.825 → 0.910 |
| 8 | 评估 | `## 评估准确率` | `model.evaluate(...)` | 测试集准确率 **86.98%** |
| 9 | 预测 | `## 预测` | 套一层 `Softmax`，再用 `np.argmax` | logits → 概率 → 取概率最大的类别 |
| 10 | 可视化 | `## 预测结果` | `plot_image()` / `plot_value_array()` | 蓝色 = 预测正确，红色 = 预测错误 |

## 模型结构

```
层                形状变化            参数量     说明
Flatten           (28,28) → (784,)    0         把二维图片拉平成一维向量
Dense(128, relu)  784 → 128           100,480   全连接隐藏层
Dense(10)         128 → 10            1,290     输出 10 个类别的 logits（无激活）
                                      101,770   合计
```

## 运行结果（本机实测）

- 环境：RTX 4060（compute capability 8.9），TensorFlow 2.21.0，CUDA 12.9 / cuDNN 9.27
- 训练 10 轮，每轮约 1~2 秒：准确率 0.825 → 0.910，loss 0.496 → 0.240
- 测试集：准确率 **86.98%**，loss 0.372

训练准确率（0.910）高于测试准确率（0.870），出现了轻微过拟合 —— 这是很好的第一个讨论点。

## 如何运行

```bash
conda activate clothing-clf     # 依赖：tensorflow 2.21.0 + numpy + matplotlib
jupyter lab                     # 打开 clothes-classifier.ipynb，按顺序执行

python app.py                   # 或只跑精简版脚本
```

### 一个容易踩的坑：检测不到 GPU

如果 `nvidia-smi` 正常，但 `tf.config.list_physical_devices('GPU')` 返回 `[]`，通常不是驱动问题，而是 **CUDA 运行库的路径问题**。

本项目的 CUDA 库来自 pip 的 `nvidia-*-cu12` 包（`tensorflow[and-cuda]` 的依赖），装在 `site-packages/nvidia/<pkg>/lib/` 下。这个目录默认不在动态链接器的搜索路径里，TensorFlow 按库名加载时会失败，日志里能看到：

```
Cannot dlopen some GPU libraries. ... Skipping registering GPU devices
```

解决办法是让这些目录能被找到，任选一种：

- 把 `site-packages/nvidia/*/lib` 加入 `LD_LIBRARY_PATH`（临时，需重启内核）
- 在 `site-packages/` 下放一个预加载脚本，导入 TensorFlow 前先 `ctypes.CDLL(..., RTLD_GLOBAL)`（持久，推荐）

## 教学要点

- **完整闭环**：从原始像素到"预测一张图"，中间只有这 8 个环节
- **归一化**：除以 255 让输入落在 0~1，梯度更稳定
- **logits vs 概率**：`from_logits=True` 时模型输出未归一化的分数，Softmax 只在需要"看概率"时才加
- **过拟合**：训练集 0.910 vs 测试集 0.870，差距就是过拟合的开始
