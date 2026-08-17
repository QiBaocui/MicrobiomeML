#!/usr/bin/env Rscript

# ============================================================
# run_model.R
# ============================================================

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  normalizePath(getwd())
}
source(file.path(script_dir, "model.R"))

library(dplyr)
library(data.table)
library(ggplot2)
library(optparse)
library(caret)
library(h2o)
library(e1071)
library(pROC)

# ------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------
option_list <- list(

  make_option(c("-o", "--outdir"),
              type = "character", default = NULL,
              help = "输出目录 [必填]"),

  make_option(c("-g", "--group"),
              type = "character", default = NULL,
              help = "分组文件路径（含ID和Group两列）[必填]"),

  make_option(c("-d", "--dat"),
              type = "character", default = NULL,
              help = "特征矩阵文件路径 [必填]"),

  make_option(c("-t", "--trainNum"),
              type = "numeric", default = 0.7,
              help = "训练集比例（proportional）或每类样本数（sample_size）[default: %default]"),

  make_option(c("-S", "--split_type"),
              type = "character", default = "proportional",
              help = "拆分方式: proportional / sample_size [default: %default]"),

  make_option(c("-p", "--ip"),
              type = "numeric", default = 54321,
              help = "H2O端口 [default: %default]"),

  # ----------------------------------------------------------
  # 核心：H2O所有模型 + SVM CV 共用同一折数，真正对齐
  # ----------------------------------------------------------
  make_option(c("-f", "--nfolds"),
              type = "integer", default = 5L,
              help = "交叉验证折数（H2O所有模型与SVM CV共用，保持一致）[default: %default]"),

  # ----------------------------------------------------------
  # SVM 相关参数
  # ----------------------------------------------------------
  make_option(c("-K", "--svm_kernel"),
              type = "character", default = "radial",
              help = "SVM核函数: radial/linear/polynomial/sigmoid [default: %default]"),

  make_option(c("-T", "--svm_tune"),
              type = "logical", default = FALSE,
              help = "是否进行SVM网格调参 [default: %default]"),

  # 调参折数独立设置，建议 < nfolds，防止内层样本不足
  make_option(c("--svm_tune_folds"),
              type = "integer", default = 3L,
              help = "SVM调参内层CV折数（独立于--nfolds，建议 < nfolds）[default: %default]"),

  make_option(c("-C", "--svm_cv"),
              type = "logical", default = TRUE,
              help = "是否对SVM进行交叉验证（折数与--nfolds一致）[default: %default]"),

  make_option(c("-m", "--svm_max_vars"),
              type = "integer", default = 300L,
              help = "SVM置换重要性最大特征数（非线性核）[default: %default]"),

  make_option(c("-n", "--svm_n_perm"),
              type = "integer", default = 5L,
              help = "SVM置换重要性置换次数 [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt        <- parse_args(opt_parser)

# ------------------------------------------------------------
# 参数校验
# ------------------------------------------------------------
if (is.null(opt$outdir)) stop("[Error] 请指定输出目录 -o/--outdir",  call. = FALSE)
if (is.null(opt$group))  stop("[Error] 请指定分组文件 -g/--group",    call. = FALSE)
if (is.null(opt$dat))    stop("[Error] 请指定特征矩阵 -d/--dat",       call. = FALSE)
if (opt$nfolds         < 2L) stop("[Error] --nfolds 必须 >= 2",        call. = FALSE)
if (opt$svm_tune_folds < 2L) stop("[Error] --svm_tune_folds 必须 >= 2",call. = FALSE)
if (opt$svm_tune_folds >= opt$nfolds)
  warning(sprintf(
    "[Warn] --svm_tune_folds(%d) >= --nfolds(%d)，建议调参折数小于CV折数，防止内层样本不足",
    opt$svm_tune_folds, opt$nfolds))

# ------------------------------------------------------------
# 变量赋值
# ------------------------------------------------------------
outdir         <- opt$outdir
split_type     <- opt$split_type
trainNum       <- opt$trainNum
IPaddress      <- opt$ip
nfolds         <- opt$nfolds          # H2O + SVM CV 共用
svm_kernel     <- opt$svm_kernel
svm_tune_cv    <- opt$svm_tune
svm_tune_folds <- opt$svm_tune_folds  # SVM调参专用（独立）
svm_cv         <- opt$svm_cv
svm_max_vars   <- opt$svm_max_vars
svm_n_perm     <- opt$svm_n_perm

if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

# ------------------------------------------------------------
# 打印运行配置
# ------------------------------------------------------------
message("============================================================")
message(sprintf("  输出目录        : %s", outdir))
message(sprintf("  拆分方式        : %s (trainNum=%.2f)", split_type, trainNum))
message(sprintf("  交叉验证折数    : %d  ← H2O所有模型 + SVM CV 共用", nfolds))
message(sprintf("  SVM核函数       : %s", svm_kernel))
message(sprintf("  SVM交叉验证     : %s  (折数与nfolds一致=%d)", svm_cv, nfolds))
message(sprintf("  SVM网格调参     : %s  (调参内层折数=%d)", svm_tune_cv, svm_tune_folds))
message(sprintf("  SVM置换重要性   : max_vars=%d  n_perm=%d", svm_max_vars, svm_n_perm))
message("============================================================")

# ------------------------------------------------------------
# 读取数据
# ------------------------------------------------------------
group <- read.table(opt$group, sep = "\t", header = TRUE,
                    stringsAsFactors = FALSE, check.names = FALSE)
dat   <- fread(opt$dat, sep = "\t", header = TRUE) %>% as.data.frame()

# ------------------------------------------------------------
# 样本拆分
# ------------------------------------------------------------
group_split <- splitSample(group      = group,
                            split_type = split_type,
                            ratio      = trainNum,
                            num        = trainNum)
train_group <- group_split$train_group
test_group  <- group_split$test_group
message(sprintf("[Data] 训练集: %d 个样本，测试集: %d 个样本",
                nrow(train_group), nrow(test_group)))

# 安全检查：折数不能超过训练集最小类别样本数
min_class_n <- min(table(train_group[, 2]))
if (nfolds > min_class_n)
  warning(sprintf(
    "[Warn] nfolds(%d) > 训练集最小类别样本数(%d)，可能导致某折只有单类！建议减小--nfolds",
    nfolds, min_class_n))

# ------------------------------------------------------------
# 特征矩阵拼接
# ------------------------------------------------------------
train <- left_join(train_group, dat, by = "ID")
test  <- left_join(test_group,  dat, by = "ID")

rownames(train) <- train[, 1]; train <- train[, -1, drop = FALSE]
rownames(test)  <- test[, 1];  test  <- test[, -1,  drop = FALSE]

# ------------------------------------------------------------
# 保存训练/测试数据
# ------------------------------------------------------------
df <- list(train_group = train_group, test_group = test_group,
           train = train, test = test)
save(df, file = paste0(outdir, "/train_data.Rdata"))
message("[Data] 训练/测试数据已保存至 ", outdir, "/train_data.Rdata")

# ------------------------------------------------------------
# 模型训练与评估
# ------------------------------------------------------------
h2o.init(ip = "localhost", port = IPaddress)

modelFun(
  train.R           = train,
  test.R            = test,
  outdir            = outdir,
  feature           = "microbe",
  feature_type_list = c("GBM", "GLM", "RF", "DL", "xGboost", "Bayes", "SVM"),
  nfolds            = nfolds,         # H2O + SVM CV 共用
  svm_kernel        = svm_kernel,
  svm_tune_cv       = svm_tune_cv,
  svm_tune_folds    = svm_tune_folds, # SVM调参专用
  svm_cv            = svm_cv,
  svm_max_vars      = svm_max_vars,
  svm_n_perm        = svm_n_perm
)

h2o.shutdown(prompt = FALSE)
message("[Done] 所有模型运行完毕！")
