# 使用指南

## 运行训练代码

要运行训练代码，请使用以下命令：

```bash
torchrun --master_port=31467 --nproc_per_node=4 trainers/train_flow.py --config configs/flowtest.toml
```

### 配置文件示例

以下是一个配置文件 `flowtest.toml` 的示例，您可以根据需要进行修改：

```toml
# 配置文件示例

# 设置批处理大小
# 默认值为100,
# batch size = batch_size * gpu counts in ddp
batch_size = 25
learning_rate = 0.0001
datastr = 'tartanair'

# useless in ddp 
# worker_num = 4

# 是否只使用光流
# 加快数据集加载速度
# Stage 1 训练用不到RGB图像
flow_only = false
rcr_type = "NORCR"
shuffle = true

# 图像的宽度和高度
image_width = 640  
image_height = 448 

total_iterations = 5000
train_path = "data/test.txt"
val_path = "data/test.txt"
pretrained_model_path = ""
save_path = "checkpoints/test"
summary_path = "logs/test"
```

## 训练脚本说明

- `torchrun`: 用于分布式训练的命令，`--nproc_per_node=4` 表示使用4个GPU。
- `--config`: 指定配置文件的路径。

## 配置项说明

- `batch_size`: 每个GPU的批处理大小。
- `learning_rate`: 学习率，控制模型更新的步长。
- `datastr`: 数据集的标识符。
- `flow_only`: 是否仅使用光流数据进行训练。
- `rcr_type`: RCR方法的类型。
- `shuffle`: 是否在每个epoch时打乱数据。
- `image_width`, `image_height`: 输入图像的尺寸。
- `total_iterations`: 总的训练迭代次数。
- `train_path`, `val_path`: 训练和验证数据的路径。
- `pretrained_model_path`: 预训练模型的路径（如果有）。
- `save_path`: 模型保存路径。
- `summary_path`: 日志保存路径。
