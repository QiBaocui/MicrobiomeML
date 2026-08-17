# ============================================================
# model.R
# ============================================================

# ------------------------------------------------------------
# 1. splitSample：样本拆分（分层）
# ------------------------------------------------------------
splitSample <- function(group,
                        split_type = "proportional",
                        ratio      = 0.7,
                        num        = 100) {
  colnames(group)[1:2] <- c("ID", "Group")

  if (split_type == "proportional") {
    library(caret)
    train_index <- createDataPartition(group[, 2], p = ratio, list = FALSE)
    train_group <- group[ train_index, , drop = FALSE]
    test_group  <- group[-train_index, , drop = FALSE]

  } else if (split_type == "sample_size") {
    per_group_num <- floor(num / 2)
    group_levels  <- unique(group$Group)
    if (length(group_levels) != 2)
      stop("split_type='sample_size' 仅支持2个分组！当前分组数：",
           length(group_levels))
    train_list <- list()
    for (i in seq_along(group_levels)) {
      sub_group <- group[group$Group == group_levels[i], , drop = FALSE]
      if (nrow(sub_group) < per_group_num)
        stop("分组 ", group_levels[i], " 样本量不足（仅",
             nrow(sub_group), "个），无法抽取", per_group_num, "个！")
      idx <- sample(seq_len(nrow(sub_group)),
                    size = per_group_num, replace = FALSE)
      train_list[[i]] <- sub_group[idx, , drop = FALSE]
    }
    train_group           <- do.call(rbind, train_list)
    test_group            <- group[!group$ID %in% train_group$ID, , drop = FALSE]
    rownames(train_group) <- NULL
    rownames(test_group)  <- NULL

  } else {
    stop("split_type 参数错误，请选择 'proportional' 或 'sample_size'")
  }

  return(list(train_group = train_group, test_group = test_group))
}


# ------------------------------------------------------------
# 2. get_outputperformance：H2O最优截点（Youden指数）
# ------------------------------------------------------------
get_outputperformance <- function(perf) {
  outputperformance.R <- as.data.frame(
    perf %>% .@metrics %>% .$thresholds_and_metric_scores
  )
  youdenIndex <- outputperformance.R$tpr - outputperformance.R$fpr
  site        <- which(youdenIndex == max(youdenIndex))[1]
  return(outputperformance.R[site, , drop = FALSE])
}


# ------------------------------------------------------------
# 3. get_predict：H2O模型预测评估（含AUC 95%CI）
# ------------------------------------------------------------
get_predict <- function(model, modelName, test.R, test.h2o, outdir, feature) {
  library(pROC); library(ggplot2); library(dplyr)

  pred_matri <- h2o.predict(object = model, newdata = test.h2o) %>%
    as.data.frame()
  pred_matri <- cbind(Group       = test.R[, 1], pred_matri)
  pred_matri <- cbind(sample_code = rownames(test.R), pred_matri)

  test_perf      <- h2o.performance(model = model, newdata = test.h2o)
  cutoffpermance <- get_outputperformance(test_perf)
  auc_val        <- h2o.auc(test_perf)

  group_levels <- sort(unique(as.character(test.R[, 1])))
  pos_class    <- group_levels[2]

  if (pos_class %in% colnames(pred_matri)) {
    prob_pos <- pred_matri[, pos_class]
  } else {
    prob_pos <- pred_matri[, ncol(pred_matri)]
    warning(sprintf("[%s-%s] 未找到正类列'%s'，使用最后一列代替",
                    feature, modelName, pos_class))
  }

  actual_bin  <- as.numeric(test.R[, 1] == pos_class)
  rocres_pROC <- roc(actual_bin, prob_pos, quiet = TRUE)
  ci_res <- tryCatch(
    ci.auc(rocres_pROC, method = "delong", conf.level = 0.95),
    error = function(e) {
      message(sprintf("[%s-%s] DeLong失败，改用bootstrap", feature, modelName))
      ci.auc(rocres_pROC, method = "bootstrap",
             boot.n = 1000, conf.level = 0.95)
    }
  )

  cutoffpermance$auc     <- auc_val
  cutoffpermance$CI_low  <- round(as.numeric(ci_res[1]), 4)
  cutoffpermance$CI_high <- round(as.numeric(ci_res[3]), 4)
  cutoffpermance$model   <- modelName

  rocData <- test_perf %>% .@metrics %>% .$thresholds_and_metric_scores %>%
    .[c("tpr", "fpr")] %>%
    add_row(tpr = 0, fpr = 0, .before = TRUE)
  rocData       <- rocData[order(rocData$fpr), , drop = FALSE]
  rocData$label <- paste0(modelName, "_AUC: ", signif(auc_val, digits = 3))

  rocCurve <- ggplot(rocData, aes(x = fpr, y = tpr, color = label)) +
    geom_line(size = 1.2) + theme_bw() +
    xlab("False positive rate") + ylab("True positive rate") +
    labs(color = "") +
    scale_color_manual(values = c("#4e639e")) +
    geom_abline(intercept = 0, slope = 1, colour = "grey") +
    ggtitle("ROC Curve") +
    theme(legend.position  = c(.75, .2),
          axis.text.x      = element_text(size = 10, face = "bold"),
          axis.text.y      = element_text(size = 10, face = "bold"),
          axis.title.x     = element_text(size = 15, face = "bold"),
          axis.title.y     = element_text(size = 15, face = "bold"),
          legend.text      = element_text(size = 11),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5),
          plot.margin      = unit(c(1, 1, 1, 1), "cm"))

  var <- tryCatch({
    if (modelName == "stacked") {
      h2o.permutation_importance(model, newdata = test.h2o) %>% as.data.frame()
    } else {
      h2o.varimp(model) %>% as.data.frame()
    }
  }, error = function(e) data.frame())

  options(bitmapType = "cairo")
  png(paste0(outdir,"/",feature,"_",modelName,"_ROC.png"),
      width = 800, height = 800)
  print(rocCurve); dev.off()
  ggsave(paste0(outdir,"/",feature,"_",modelName,"_ROC.pdf"),
         rocCurve, width = 8, height = 8)
  write.table(pred_matri,
              paste0(outdir,"/",feature,"_",modelName,"_pred.tsv"),
              sep="\t", quote=F, row.names=F, col.names=T, na="")
  write.table(cutoffpermance,
              paste0(outdir,"/",feature,"_",modelName,"_cutoffpermance.tsv"),
              sep="\t", quote=F, row.names=F, col.names=T, na="")
  write.table(rocData,
              paste0(outdir,"/",feature,"_",modelName,"_rocData.tsv"),
              sep="\t", quote=F, row.names=F, col.names=T, na="")
  write.table(var,
              paste0(outdir,"/",feature,"_",modelName,"_varimp.tsv"),
              sep="\t", quote=F, row.names=F, col.names=T, na="")

  message(sprintf("[H2O] %s | %s | AUC=%.4f [%.4f-%.4f]",
                  feature, modelName, auc_val, ci_res[1], ci_res[3]))
}


# ------------------------------------------------------------
# 4. train_svm：SVM模型训练
#    tune_folds：调参内层CV折数，由外部传入，不写死
# ------------------------------------------------------------
train_svm <- function(train.R,
                      kernel     = "radial",
                      tune_cv    = FALSE,
                      tune_folds = 3L) {
  library(e1071)

  train_data       <- train.R
  train_data$Group <- as.factor(train_data$Group)
  gamma_default    <- 1 / max(ncol(train_data) - 1, 1)

  if (tune_cv) {
    message(sprintf("[SVM] 网格调参（%d折内层CV, kernel=%s）...",
                    tune_folds, kernel))
    if (kernel == "linear") {
      tuned <- tune.svm(
        Group ~ ., data = train_data,
        kernel      = kernel,
        cost        = c(0.1, 1, 10, 100),
        tunecontrol = tune.control(cross = tune_folds)
      )
    } else {
      tuned <- tune.svm(
        Group ~ ., data = train_data,
        kernel      = kernel,
        cost        = c(0.1, 1, 10, 100),
        gamma       = c(gamma_default/10, gamma_default, gamma_default*10),
        tunecontrol = tune.control(cross = tune_folds)
      )
    }
    best_cost  <- tuned$best.parameters$cost
    best_gamma <- if (!is.null(tuned$best.parameters$gamma))
      tuned$best.parameters$gamma else gamma_default
    message(sprintf("[SVM] 最优参数: cost=%.4f, gamma=%.6f",
                    best_cost, best_gamma))
  } else {
    best_cost  <- 1
    best_gamma <- gamma_default
    message(sprintf("[SVM] 默认参数: cost=%.4f, gamma=%.6f",
                    best_cost, best_gamma))
  }

  if (kernel == "linear") {
    svm_model <- svm(Group ~ ., data = train_data,
                     kernel = kernel, cost = best_cost,
                     probability = TRUE)
  } else {
    svm_model <- svm(Group ~ ., data = train_data,
                     kernel = kernel, cost = best_cost,
                     gamma  = best_gamma, probability = TRUE)
  }
  return(svm_model)
}


# ------------------------------------------------------------
# 5. cv_svm：SVM分层K折交叉验证
#
#    与H2O完全对齐的四个关键点：
#    ① cv_folds   直接传入 nfolds（与H2O共用，不单独设参数）
#    ② 分层抽样   createFolds() 对应 fold_assignment="Stratified"
#    ③ 随机种子   set.seed(1)   对应 H2O seed=1
#    ④ tune_folds 独立参数，调参专用，不与cv_folds混用
# ------------------------------------------------------------
cv_svm <- function(train.R,
                   kernel     = "radial",
                   tune_cv    = FALSE,
                   tune_folds = 3L,
                   cv_folds   = 5L,    # 由外部传入nfolds，与H2O共用
                   outdir,
                   feature) {
  library(caret); library(pROC); library(e1071)

  group_levels <- sort(unique(as.character(train.R[, 1])))
  if (length(group_levels) != 2)
    stop("[SVM CV] 仅支持二分类，当前分组数：", length(group_levels))
  pos_class <- group_levels[2]
  neg_class <- group_levels[1]

  message(sprintf(
    "[SVM CV] %s | %d折分层CV | kernel=%s | tune_cv=%s | tune_folds=%d",
    feature, cv_folds, kernel, tune_cv, tune_folds))
  message(sprintf(
    "[SVM CV] 正类='%s' | 对齐H2O: fold_assignment='Stratified' + seed=1",
    pos_class))

  # 固定随机种子，与H2O的 seed=1 对齐
  set.seed(1)
  fold_indices <- createFolds(as.factor(train.R[, 1]),
                              k           = cv_folds,
                              list        = TRUE,
                              returnTrain = FALSE)

  cv_results <- vector("list", cv_folds)

  for (fold_i in seq_len(cv_folds)) {
    message(sprintf("[SVM CV] --- Fold %d / %d ---", fold_i, cv_folds))

    val_idx    <- fold_indices[[fold_i]]
    train_fold <- train.R[-val_idx, , drop = FALSE]
    val_fold   <- train.R[ val_idx, , drop = FALSE]

    if (length(unique(train_fold[, 1])) < 2) {
      warning(sprintf("[SVM CV] Fold %d 训练折含单类，跳过！", fold_i)); next
    }
    if (length(unique(val_fold[, 1])) < 2) {
      warning(sprintf("[SVM CV] Fold %d 验证折含单类，跳过！", fold_i)); next
    }

    fold_model <- tryCatch(
      train_svm(train.R    = train_fold,
                kernel     = kernel,
                tune_cv    = tune_cv,
                tune_folds = tune_folds),
      error = function(e) {
        warning(sprintf("[SVM CV] Fold %d 训练失败：%s", fold_i, e$message))
        NULL
      }
    )
    if (is.null(fold_model)) next

    pred_raw  <- predict(fold_model,
                         newdata     = val_fold[, -1, drop = FALSE],
                         probability = TRUE)
    pred_prob <- attr(pred_raw, "probabilities")

    if (!pos_class %in% colnames(pred_prob)) {
      warning(sprintf("[SVM CV] Fold %d 正类列'%s'缺失，跳过！",
                      fold_i, pos_class)); next
    }

    prob_pos   <- pred_prob[, pos_class]
    actual_bin <- as.numeric(val_fold[, 1] == pos_class)
    true_label <- as.character(val_fold[, 1])

    fold_auc <- tryCatch(
      as.numeric(auc(roc(actual_bin, prob_pos, quiet = TRUE))),
      error = function(e) NA_real_
    )

    rocres_fold <- tryCatch(
      roc(actual_bin, prob_pos, quiet = TRUE),
      error = function(e) NULL
    )

    if (!is.null(rocres_fold) && length(rocres_fold$thresholds) > 1) {
      youden    <- rocres_fold$sensitivities + rocres_fold$specificities - 1
      best_idx  <- which.max(youden)
      best_thr  <- rocres_fold$thresholds[best_idx]

      pred_class <- ifelse(prob_pos >= best_thr, pos_class, neg_class)
      TP <- sum(pred_class == pos_class & true_label == pos_class)
      FP <- sum(pred_class == pos_class & true_label != pos_class)
      FN <- sum(pred_class != pos_class & true_label == pos_class)
      TN <- sum(pred_class != pos_class & true_label != pos_class)

      precision <- ifelse((TP+FP)>0, TP/(TP+FP), NA_real_)
      recall    <- ifelse((TP+FN)>0, TP/(TP+FN), NA_real_)
      F1        <- ifelse(!is.na(precision) & !is.na(recall) &
                            (precision+recall) > 0,
                          2*precision*recall/(precision+recall), NA_real_)
      accuracy  <- mean(pred_class == true_label)
      tpr       <- rocres_fold$sensitivities[best_idx]
      tnr       <- rocres_fold$specificities[best_idx]
      threshold <- best_thr
    } else {
      TP<-FP<-FN<-TN        <- NA_integer_
      precision<-recall<-F1 <- NA_real_
      accuracy<-tpr<-tnr    <- NA_real_
      threshold              <- NA_real_
    }

    cv_results[[fold_i]] <- data.frame(
      Fold      = fold_i,
      N_val     = nrow(val_fold),
      threshold = round(threshold, 6),
      AUC       = round(fold_auc,  4),
      Accuracy  = round(accuracy,  4),
      TPR       = round(tpr,       4),
      TNR       = round(tnr,       4),
      Precision = round(precision, 4),
      F1        = round(F1,        4),
      TP=TP, FP=FP, FN=FN, TN=TN,
      stringsAsFactors = FALSE
    )

    message(sprintf(
      "[SVM CV] Fold %d | N=%d | AUC=%.4f | Acc=%.4f | TPR=%.4f | TNR=%.4f | Pre=%.4f | F1=%.4f",
      fold_i, nrow(val_fold),
      ifelse(is.na(fold_auc),  0, fold_auc),
      ifelse(is.na(accuracy),  0, accuracy),
      ifelse(is.na(tpr),       0, tpr),
      ifelse(is.na(tnr),       0, tnr),
      ifelse(is.na(precision), 0, precision),
      ifelse(is.na(F1),        0, F1)
    ))
  }

  valid_results <- Filter(Negate(is.null), cv_results)
  if (length(valid_results) == 0) {
    warning("[SVM CV] 所有折均失败！请检查数据量和类别分布！")
    return(invisible(NULL))
  }

  cv_df    <- do.call(rbind, valid_results)
  num_cols <- c("threshold","AUC","Accuracy","TPR","TNR","Precision","F1")

  # Mean与SD分两行，便于下游程序解析
  mean_row <- data.frame(
    Fold="Mean", N_val=NA,
    as.list(round(colMeans(cv_df[, num_cols], na.rm=TRUE), 4)),
    TP=NA, FP=NA, FN=NA, TN=NA, stringsAsFactors=FALSE
  )
  sd_row <- data.frame(
    Fold="SD", N_val=NA,
    as.list(round(apply(cv_df[, num_cols], 2, sd, na.rm=TRUE), 4)),
    TP=NA, FP=NA, FN=NA, TN=NA, stringsAsFactors=FALSE
  )

  output_df <- rbind(
    lapply(cv_df, as.character) %>% as.data.frame(stringsAsFactors=FALSE),
    mean_row,
    sd_row
  )

  cv_out <- paste0(outdir, "/", feature, "_SVM_cv_results.tsv")
  write.table(output_df, cv_out,
              sep="\t", quote=FALSE, row.names=FALSE,
              col.names=TRUE, na="")

  message(sprintf(
    "[SVM CV] %s | 完成（有效%d/%d折）| AUC=%.4f±%.4f | F1=%.4f±%.4f",
    feature, nrow(cv_df), cv_folds,
    mean(cv_df$AUC, na.rm=TRUE), sd(cv_df$AUC, na.rm=TRUE),
    mean(cv_df$F1,  na.rm=TRUE), sd(cv_df$F1,  na.rm=TRUE)
  ))
  message("[SVM CV] 结果已保存至: ", cv_out)

  return(invisible(cv_df))
}


# ------------------------------------------------------------
# 6. get_varimp_svm：SVM特征重要性
#    linear核  → 权重向量法（精确快速）
#    非linear核 → 置换重要性（AUC下降量）
# ------------------------------------------------------------
get_varimp_svm <- function(model, test.R, pos_class,
                            max_vars = 300L, n_perm = 5L) {
  library(e1071); library(pROC)

  kernel_type <- model$kernel

  if (kernel_type == "linear") {
    message("[SVM VarImp] 线性核：权重向量法")
    w <- tryCatch(
      t(model$coefs) %*% model$SV,
      error = function(e) {
        warning("[SVM VarImp] 权重提取失败，返回空结果"); NULL
      }
    )
    if (is.null(w)) return(data.frame())

    w_abs        <- abs(as.vector(w))
    names(w_abs) <- colnames(model$SV)
    var_df       <- data.frame(Variable           = names(w_abs),
                                Relative_Importance = w_abs,
                                stringsAsFactors    = FALSE)
    var_df       <- var_df[order(var_df$Relative_Importance,
                                  decreasing = TRUE), ]
    max_imp                  <- max(var_df$Relative_Importance, 1e-10)
    var_df$Scaled_Importance <- var_df$Relative_Importance / max_imp
    var_df$Percentage        <- var_df$Relative_Importance /
      sum(var_df$Relative_Importance) * 100
    var_df$Method            <- "weight_vector"
    rownames(var_df)         <- NULL
    return(var_df)
  }

  message(sprintf("[SVM VarImp] %s核：置换重要性（n_perm=%d）",
                  kernel_type, n_perm))

  X        <- test.R[, -1, drop = FALSE]
  y_factor <- as.factor(test.R[, 1])

  base_prob <- attr(predict(model, newdata=X, probability=TRUE),
                    "probabilities")[, pos_class]
  base_auc  <- suppressMessages(
    as.numeric(auc(roc(y_factor, base_prob, quiet=TRUE)))
  )

  feat_names <- colnames(X)
  sampled    <- FALSE
  if (length(feat_names) > max_vars) {
    message(sprintf("[SVM VarImp] 特征数%d > max_vars=%d，随机抽取%d个",
                    length(feat_names), max_vars, max_vars))
    feat_names <- sample(feat_names, max_vars)
    sampled    <- TRUE
  }

  imp_vals <- sapply(feat_names, function(feat) {
    mean(replicate(n_perm, {
      X_p         <- X
      X_p[, feat] <- sample(X_p[, feat])
      perm_prob   <- attr(predict(model, newdata=X_p, probability=TRUE),
                          "probabilities")[, pos_class]
      tryCatch(
        base_auc - suppressMessages(
          as.numeric(auc(roc(y_factor, perm_prob, quiet=TRUE)))
        ),
        error = function(e) 0
      )
    }))
  })

  var_df    <- data.frame(Variable           = feat_names,
                           Relative_Importance = imp_vals,
                           stringsAsFactors    = FALSE)
  var_df    <- var_df[order(var_df$Relative_Importance, decreasing=TRUE), ]
  max_imp   <- max(abs(var_df$Relative_Importance), 1e-10)
  pos_sum   <- sum(var_df$Relative_Importance[var_df$Relative_Importance > 0])

  var_df$Scaled_Importance <- var_df$Relative_Importance / max_imp
  var_df$Percentage        <- ifelse(pos_sum > 0,
                                      var_df$Relative_Importance/pos_sum*100, 0)
  var_df$Method            <- "permutation_AUC_drop"
  var_df$Sampled           <- sampled
  rownames(var_df)         <- NULL
  return(var_df)
}


# ------------------------------------------------------------
# 7. get_predict_svm：SVM独立测试集评估
# ------------------------------------------------------------
get_predict_svm <- function(model, modelName = "SVM",
                             train.R, test.R,
                             outdir, feature,
                             max_vars = 300L, n_perm = 5L) {
  library(e1071); library(pROC); library(ggplot2); library(dplyr)

  group_levels <- sort(unique(as.character(train.R[, 1])))
  pos_class    <- group_levels[2]
  neg_class    <- group_levels[1]

  pred_raw  <- predict(model, newdata=test.R[,-1,drop=FALSE],
                       probability=TRUE)
  pred_prob <- attr(pred_raw, "probabilities")

  pred_matri <- data.frame(
    sample_code = rownames(test.R),
    Group       = test.R[, 1],
    predict     = as.character(pred_raw),
    stringsAsFactors = FALSE
  )
  for (cls in group_levels)
    if (cls %in% colnames(pred_prob))
      pred_matri[[cls]] <- pred_prob[, cls]

  if (!pos_class %in% colnames(pred_prob))
    stop(sprintf("[SVM] 正类'%s'不在预测概率矩阵中，请检查标签", pos_class))

  actual_bin <- as.numeric(test.R[, 1] == pos_class)
  prob_pos   <- pred_prob[, pos_class]
  rocres     <- roc(actual_bin, prob_pos, quiet=TRUE)
  auc_val    <- as.numeric(auc(rocres))

  ci_res <- tryCatch(
    ci.auc(rocres, method="delong", conf.level=0.95),
    error = function(e) {
      message("[SVM] DeLong失败，改用bootstrap（n=1000）")
      ci.auc(rocres, method="bootstrap", boot.n=1000, conf.level=0.95)
    }
  )

  youden         <- rocres$sensitivities + rocres$specificities - 1
  best_idx       <- which.max(youden)
  best_threshold <- rocres$thresholds[best_idx]
  true_label     <- as.character(test.R[, 1])
  pred_class     <- ifelse(prob_pos >= best_threshold, pos_class, neg_class)

  TP <- sum(pred_class == pos_class & true_label == pos_class)
  FP <- sum(pred_class == pos_class & true_label != pos_class)
  FN <- sum(pred_class != pos_class & true_label == pos_class)
  TN <- sum(pred_class != pos_class & true_label != pos_class)

  precision <- ifelse((TP+FP)>0, TP/(TP+FP), NA_real_)
  recall    <- ifelse((TP+FN)>0, TP/(TP+FN), NA_real_)
  F1_score  <- ifelse(!is.na(precision) & !is.na(recall) &
                        (precision+recall) > 0,
                      2*precision*recall/(precision+recall), NA_real_)

  cutoffpermance <- data.frame(
    threshold = round(best_threshold, 6),
    accuracy  = round(mean(pred_class == true_label), 4),
    tpr       = round(rocres$sensitivities[best_idx], 4),
    fpr       = round(1 - rocres$specificities[best_idx], 4),
    tnr       = round(rocres$specificities[best_idx], 4),
    precision = round(precision, 4),
    F1        = round(F1_score,  4),
    TP=TP, FP=FP, FN=FN, TN=TN,
    auc       = round(auc_val, 4),
    CI_low    = round(as.numeric(ci_res[1]), 4),
    CI_high   = round(as.numeric(ci_res[3]), 4),
    model     = modelName,
    stringsAsFactors = FALSE
  )

  rocData <- data.frame(fpr = 1 - rocres$specificities,
                         tpr = rocres$sensitivities)
  #rocData       <- rocData[order(rocData$fpr), ]
  rocData <- rocData %>% arrange(fpr, tpr)
  rocData       <- add_row(rocData, tpr=0, fpr=0, .before=TRUE)
  rocData$label <- paste0(modelName, "_AUC: ", signif(auc_val, digits=3))

  rocCurve <- ggplot(rocData, aes(x=fpr, y=tpr, color=label)) +
    geom_line(size=1.2) + theme_bw() +
    xlab("False positive rate") + ylab("True positive rate") +
    labs(color="") +
    scale_color_manual(values=c("#4e639e")) +
    geom_abline(intercept=0, slope=1, colour="grey") +
    ggtitle("ROC Curve") +
    theme(legend.position  = c(.75, .2),
          axis.text.x      = element_text(size=10, face="bold"),
          axis.text.y      = element_text(size=10, face="bold"),
          axis.title.x     = element_text(size=15, face="bold"),
          axis.title.y     = element_text(size=15, face="bold"),
          legend.text      = element_text(size=11),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust=0.5),
          plot.margin      = unit(c(1,1,1,1),"cm"))

  message(sprintf("[SVM VarImp] %s | %s | 开始计算...", feature, modelName))
  var_imp <- get_varimp_svm(model     = model,
                             test.R    = test.R,
                             pos_class = pos_class,
                             max_vars  = max_vars,
                             n_perm    = n_perm)

  options(bitmapType="cairo")
  png(paste0(outdir,"/",feature,"_",modelName,"_ROC.png"),
      width=800, height=800)
  print(rocCurve); dev.off()
  ggsave(paste0(outdir,"/",feature,"_",modelName,"_ROC.pdf"),
         rocCurve, width=8, height=8)
  write.table(pred_matri,
              paste0(outdir,"/",feature,"_",modelName,"_pred.tsv"),
              sep="\t",quote=F,row.names=F,col.names=T,na="")
  write.table(cutoffpermance,
              paste0(outdir,"/",feature,"_",modelName,"_cutoffpermance.tsv"),
              sep="\t",quote=F,row.names=F,col.names=T,na="")
  write.table(rocData,
              paste0(outdir,"/",feature,"_",modelName,"_rocData.tsv"),
              sep="\t",quote=F,row.names=F,col.names=T,na="")
  write.table(var_imp,
              paste0(outdir,"/",feature,"_",modelName,"_varimp.tsv"),
              sep="\t",quote=F,row.names=F,col.names=T,na="")

  message(sprintf(
    "[SVM] %s | %s | AUC=%.4f [%.4f-%.4f] | Pre=%.4f | F1=%.4f | TP=%d FP=%d FN=%d TN=%d",
    feature, modelName, auc_val, ci_res[1], ci_res[3],
    ifelse(is.na(precision), 0, precision),
    ifelse(is.na(F1_score),  0, F1_score),
    TP, FP, FN, TN
  ))
}


# ------------------------------------------------------------
# 8. modelFun：主流程
#
#    参数设计原则（与H2O保持一致）：
#    nfolds         → H2O所有模型CV折数 + SVM外层CV折数（共用，真正对齐）
#    svm_tune_folds → SVM调参内层专用（独立，建议 < nfolds）
# ------------------------------------------------------------
modelFun <- function(train.R, test.R,
                     outdir, feature,
                     feature_type_list,
                     nfolds         = 5L,     # H2O + SVM CV 共用
                     svm_kernel     = "radial",
                     svm_tune_cv    = FALSE,
                     svm_tune_folds = 3L,      # SVM调参专用（独立）
                     svm_cv         = TRUE,
                     svm_max_vars   = 300L,
                     svm_n_perm     = 5L) {

  message("============================================================")
  message(sprintf("[modelFun] H2O CV折数  : nfolds=%d", nfolds))
  message(sprintf("[modelFun] SVM CV      : svm_cv=%s  折数=nfolds=%d（与H2O一致）",
                  svm_cv, nfolds))
  message(sprintf("[modelFun] SVM调参     : svm_tune_cv=%s  内层折数=%d（独立参数）",
                  svm_tune_cv, svm_tune_folds))
  message("============================================================")

  # =================== H2O 模型部分 ===================
  train.h2o       <- as.h2o(train.R)
  train.h2o$Group <- h2o.asfactor(train.h2o$Group)
  test.h2o        <- as.h2o(test.R)
  test.h2o$Group  <- h2o.asfactor(test.h2o$Group)

  y          <- "Group"
  x          <- setdiff(names(train.R), y)
  model_list <- list()

  if ("GBM" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 GBM (nfolds=%d)...", feature, nfolds))
    model_list[["GBM"]] <- h2o.gbm(
      x=x, y=y, training_frame=train.h2o,
      nfolds                                = nfolds,
      keep_cross_validation_predictions     = TRUE,
      fold_assignment                       = "Stratified",
      keep_cross_validation_fold_assignment = TRUE,
      seed = 1)
    get_predict(model_list[["GBM"]],"GBM",test.R,test.h2o,outdir,feature)
  }

  if ("GLM" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 GLM (nfolds=%d)...", feature, nfolds))
    model_list[["GLM"]] <- h2o.glm(
      x=x, y=y, training_frame=train.h2o,
      nfolds                                = nfolds,
      keep_cross_validation_predictions     = TRUE,
      fold_assignment                       = "Stratified",
      keep_cross_validation_fold_assignment = TRUE,
      seed = 1)
    get_predict(model_list[["GLM"]],"GLM",test.R,test.h2o,outdir,feature)
  }

  if ("RF" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 RF (nfolds=%d)...", feature, nfolds))
    model_list[["RF"]] <- h2o.randomForest(
      x=x, y=y, training_frame=train.h2o,
      nfolds                            = nfolds,
      keep_cross_validation_predictions = TRUE,
      fold_assignment                   = "Stratified",
      seed = 1)
    get_predict(model_list[["RF"]],"RF",test.R,test.h2o,outdir,feature)
  }

  if ("DL" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 DL (nfolds=%d)...", feature, nfolds))
    model_list[["DL"]] <- h2o.deeplearning(
      x=x, y=y, training_frame=train.h2o,
      nfolds                            = nfolds,
      keep_cross_validation_predictions = TRUE,
      fold_assignment                   = "Stratified",
      seed = 1)
    get_predict(model_list[["DL"]],"DL",test.R,test.h2o,outdir,feature)
  }

  if ("xGboost" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 xGboost (nfolds=%d)...", feature, nfolds))
    model_list[["xGboost"]] <- h2o.xgboost(
      x=x, y=y, training_frame=train.h2o,
      nfolds                            = nfolds,
      keep_cross_validation_predictions = TRUE,
      fold_assignment                   = "Stratified",
      seed = 1)
    get_predict(model_list[["xGboost"]],"xGboost",test.R,test.h2o,outdir,feature)
  }

  if ("Bayes" %in% feature_type_list) {
    message(sprintf("[H2O] %s | 训练 NaiveBayes (nfolds=%d)...", feature, nfolds))
    model_list[["Bayes"]] <- h2o.naiveBayes(
      x=x, y=y, training_frame=train.h2o,
      nfolds                            = nfolds,
      keep_cross_validation_predictions = TRUE,
      fold_assignment                   = "Stratified",
      seed = 1)
    get_predict(model_list[["Bayes"]],"Bayes",test.R,test.h2o,outdir,feature)
  }

  if (length(model_list) > 0) {
    message(sprintf("[H2O] %s | 训练 Stacked Ensemble...", feature))
    stacked <- h2o.stackedEnsemble(x=x, y=y,
                                    training_frame=train.h2o,
                                    base_models=model_list)
    get_predict(stacked,"stacked",test.R,test.h2o,outdir,feature)
    h2o.saveModel(stacked, path=outdir,
                  filename=paste0(feature,"_stacked_model"), force=TRUE)
    for (i in names(model_list))
      h2o.saveModel(model_list[[i]], path=outdir,
                    filename=paste0(feature,"_",i,"_model"), force=TRUE)
  }

  # =================== SVM 独立模块 ===================
  if ("SVM" %in% feature_type_list) {

    # Step 1：分层K折CV（cv_folds = nfolds，与H2O严格对齐）
    if (svm_cv) {
      message(sprintf(
        "[SVM] %s | Step 1/2 - %d折分层CV（= H2O nfolds，tune_folds=%d）...",
        feature, nfolds, svm_tune_folds))
      cv_svm(
        train.R    = train.R,
        kernel     = svm_kernel,
        tune_cv    = svm_tune_cv,
        tune_folds = svm_tune_folds,  # 调参专用折数
        cv_folds   = nfolds,          # 与H2O共用nfolds，真正对齐
        outdir     = outdir,
        feature    = feature
      )
    } else {
      message(sprintf("[SVM] %s | 跳过CV（--svm_cv FALSE）", feature))
    }

    # Step 2：全量训练集训练最终模型 → 独立测试集评估
    message(sprintf(
      "[SVM] %s | Step 2/2 - 全量训练集训练最终模型（kernel=%s, tune_cv=%s）...",
      feature, svm_kernel, svm_tune_cv))
    svm_model <- train_svm(
      train.R    = train.R,
      kernel     = svm_kernel,
      tune_cv    = svm_tune_cv,
      tune_folds = svm_tune_folds
    )

    get_predict_svm(
      model     = svm_model, modelName = "SVM",
      train.R   = train.R,   test.R    = test.R,
      outdir    = outdir,    feature   = feature,
      max_vars  = svm_max_vars, n_perm = svm_n_perm
    )

    saveRDS(svm_model,
            file = paste0(outdir, "/", feature, "_SVM_model.rds"))
    message(sprintf("[SVM] %s | 模型已保存至 %s/%s_SVM_model.rds",
                    feature, outdir, feature))
  }

  message(sprintf("===== %s 全部模型完成 =====\n", feature))
}
