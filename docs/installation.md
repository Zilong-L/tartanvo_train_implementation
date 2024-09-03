# 安装指南

## 1. 克隆仓库

首先，克隆本项目的代码仓库：

```bash
git clone git@github.com:Zilong-L/tartanvo_train_implementation.git
```

## 2. 创建Conda环境

通过以下命令创建并激活Conda环境：

```bash
conda env create -f environment.yml
conda activate my_env
```

请将 `my_env` 替换为 `environment.yml` 文件中定义的实际环境名称。

## 3. 安装依赖包

安装所需的Python依赖包：

```bash
pip install -r requirements.txt
```

## 4. 配置数据集

将您的数据集放在`data/`目录下，确保目录结构与示例保持一致。您可以使用以下脚本生成数据集的位姿文件：

```bash
python utils/list_files.py
```

生成的位姿文件将保存在 `pose_files.txt` 中。
