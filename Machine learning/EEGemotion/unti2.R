suppressMessages({
  library(tidyverse)
  library(data.table)
  library(openxlsx)
  library(seqinr)
  library(plyr)
  library(randomForestSRC)
  library(glmnet)
  library(plsRglm)
  library(gbm)
  library(caret)
  library(mboost)
  library(e1071)
  library(BART)
  library(MASS)
  library(snowfall)
  library(xgboost)
  library(RColorBrewer)
  library(pROC)
})

#### 读取数据 ####
Train_class <- fread("./Training_class.txt")
Train_expr <- fread("./Training_expr.txt") %>% column_to_rownames(var = "ID")
Train_class <- Train_class %>% dplyr::filter(sample %in% colnames(Train_expr)) %>% column_to_rownames(var = "sample")
Train_expr <- Train_expr %>% dplyr::select(rownames(Train_class)) %>% t() %>% as.data.frame()

Train_expr <- Train_expr %>% as.matrix()

## 设置结局列
train_outcome <- "outcome"
test_outcome <- "outcome"

## 按队列对数据分别进行标准化
Train_set <- scale(Train_expr, center = TRUE, scale = TRUE)

#### 加载函数 ####
source("./functions/functions.R")

#### 训练模型 ####
model.train <- list()
set.seed(123456)
mt.methods <- data.frame(
  tar2.method = c(rep("Enet", 11), rep("Stepglm", 3), "SVM", "glmBoost", "LDA", "plsRglm", "RF", "GBM", "XGBoost", "NaiveBayes"),
  tar2.args = c(seq(0, 1, 0.1), "backward", "forward", "both", rep(NA, 8))
)

two.stage.method <- data.frame(tar1 = "simple") %>%
  purrr::pmap_df(~mt.methods %>% dplyr::mutate(tar1.method.name = (...))) %>%
  dplyr::mutate(
    tar1.method = str_split(tar1.method.name, "\\[", simplify = TRUE)[, 1],
    tar2.method.name = ifelse(is.na(tar2.args), tar2.method,
                              ifelse(tar2.method == "Enet", str_c(tar2.method, "[alpha=", tar2.args, "]"),
                                     ifelse(tar2.method == "Stepglm", str_c(tar2.method, "[direction=", tar2.args, "]"), tar2.method)
                              )
    ),
    tar2.method.name = ifelse(tar2.method.name == "Enet[alpha=1]", "Lasso",
                              ifelse(tar2.method.name == "Enet[alpha=0]", "Ridge", tar2.method.name)
    ),
    method.name = tar2.method.name
  )

# 查看所有组合
print(two.stage.method)

# 进行模型训练
for (i in 1:nrow(two.stage.method)) {
  method <- two.stage.method$method.name[i]
  print(paste("Training model:", method))
  
  # 调用RunML函数进行训练
  model.train[[method]] <- RunML(
    method = two.stage.method$tar2.method[i],
    method_param = two.stage.method$tar2.args[i],
    Train_set = Train_set,
    Train_label = Train_class,
    classVar = train_outcome
  )
}
