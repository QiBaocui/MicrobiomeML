# Microbiome machine-learning workflow

该项目是一个面向二分类微生物组特征数据的 R 建模流程。它会分层拆分训练集和测试集，并训练 GBM、GLM、随机森林、深度学习、XGBoost、朴素贝叶斯、H2O Stacked Ensemble 和 SVM 模型。流程同时输出预测结果、ROC/AUC、95% 置信区间、最佳阈值、变量重要性和 SVM 交叉验证结果。

## 文件说明

- `model_cmd.R`：批量命令生成器，默认生成 10 次建模任务到 `predict.sh`。
- `run_model.R`：单次建模的命令行入口，负责参数解析、数据读取、训练/测试集拆分和启动 H2O。
- `model.R`：样本拆分、模型训练、预测评估、SVM 交叉验证及变量重要性函数。

这三个脚本应放在同一目录。脚本之间使用相对路径，不依赖原服务器上的代码目录。

## 环境依赖

- R（建议 4.1 或更高版本）
- Java（版本需与所安装的 H2O R 包兼容）
- R 包：`optparse`、`dplyr`、`data.table`、`ggplot2`、`caret`、`h2o`、`e1071`、`pROC`

可在 R 中安装 CRAN 依赖：

```r
install.packages(c(
  "optparse", "dplyr", "data.table", "ggplot2",
  "caret", "e1071", "pROC"
))
```

H2O 请按其官方 R 安装说明安装与本机 Java 兼容的版本。

## 输入文件

两个输入文件均为制表符分隔（TSV），且首行是列名。

分组文件至少包含两列；前两列会被解释为 `ID` 和 `Group`：

```text
ID\tGroup
sample_01\tControl
sample_02\tCase
```

特征矩阵必须包含 `ID` 列，其余列为数值特征：

```text
ID\tfeature_1\tfeature_2
sample_01\t0.12\t3.4
sample_02\t0.08\t1.7
```

当前评估和 SVM 流程按二分类任务设计。两个文件中的样本 ID 应一致。

## 快速开始

### 1. 生成 10 次重复建模命令

```bash
Rscript model_cmd.R \
  -o /path/to/results/species \
  -g /path/to/group.txt \
  -d /path/to/species_abun.tsv
```

该命令在输出目录中生成可执行的 `predict.sh`，并自动复用生成命令时所在 R 环境的 `Rscript`。检查内容后可运行：

```bash
bash /path/to/results/species/predict.sh
```

`predict.sh` 默认参数与原分析一致：训练集比例 0.8、5 折交叉验证、radial SVM、不进行 SVM 网格调参、3 折内层调参设置、最多评估 3000 个 SVM 特征、每个特征置换 5 次。每次任务随机选择一个 54321–65535 范围内的 H2O 端口。

### 2. 直接运行一次模型

```bash
Rscript run_model.R \
  -o /path/to/results/run1 \
  -g /path/to/group.txt \
  -d /path/to/species_abun.tsv \
  -t 0.8 \
  -S proportional \
  -f 5 \
  -K radial \
  -T FALSE \
  -C TRUE \
  --svm_tune_folds 3 \
  -m 3000 \
  -n 5 \
  -p 54321
```

查看完整参数帮助：

```bash
Rscript run_model.R --help
```

## 主要输出

每个运行目录主要包含：

- `train_data.Rdata`：训练集、测试集及对应分组。
- `microbe_<模型>_pred.tsv`：独立测试集预测。
- `microbe_<模型>_cutoffpermance.tsv`：最佳阈值和性能指标。
- `microbe_<模型>_rocData.tsv`、`*.png`、`*.pdf`：ROC 数据和图。
- `microbe_<模型>_varimp.tsv`：变量重要性。
- `microbe_SVM_cv_results.tsv`：SVM 分层交叉验证结果。
- H2O 模型文件。

## 注意事项

- H2O 需要可用的 Java 环境和空闲端口；并行运行多个任务时，每个任务必须使用不同端口。
- 特征较多时，SVM 置换重要性计算可能耗时较长，可通过 `-m` 和 `-n` 控制规模。
- 为保证可复现性，模型内部交叉验证使用固定随机种子；批量脚本的训练/测试拆分仍会在每次运行时重新抽样。
- 输出文件已加入 `.gitignore`，避免误将大体积模型和分析结果提交到 GitHub。
