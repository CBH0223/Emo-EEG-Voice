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
  #library(ComplexHeatmap)
  library(RColorBrewer)
  library(pROC)
})

#### 读取数据 ####
Train_class <- fread("./Training_class.txt")
Train_expr <- fread("./Training_expr.txt") %>% column_to_rownames(var = "ID")
Train_class <- Train_class %>% dplyr::filter(sample%in%colnames(Train_expr)) %>% column_to_rownames(var = "sample")
Train_expr <- Train_expr %>% dplyr::select(rownames(Train_class)) %>% t() %>% as.data.frame()

Test_class <- fread("./Testing_class.txt")
Test_expr <- fread("./Testing_expr.txt") %>% column_to_rownames(var = "V1")
Test_class <- Test_class %>% dplyr::filter(Sample%in%colnames(Test_expr)) %>% column_to_rownames(var = "Sample")
Test_expr <- Test_expr %>% dplyr::select(rownames(Test_class)) %>% t() %>% as.data.frame()

Train_expr <- Train_expr %>% 
  dplyr::select_at(colnames(Test_expr)) %>% 
  as.matrix()
Train_expr <- Train_expr %>% as.matrix()

Test_expr <- Test_expr %>% 
  dplyr::select_at(colnames(Train_expr)) %>% 
  as.matrix()
## 设置结局列
train_outcome <- "outcome"
test_outcome <- "outcome"
## 按队列对数据分别进行标准化(注意如果是多个数据集合并的结果视情况需要分开进行scale)
Train_set <- scale(Train_expr, center = T, scale = T) 
Test_set <- scale(Test_expr, center = T, scale = T)

#### 加载函数 ####
source("./functions/functions.R")

#### 特征筛选 ####
## 算法包括：Lasso、Ridge、Enet、Stepglm、SVM、glmBoost、LDA、plsRglm、RandomForest、GBM、XGBoost、NaiveBayes
## 可以进行特征提取的算法有：Lasso、Ridge、Enet、Stepglm、glmBoost、plsRglm、RF、GBM，我们使用这8种方法进行特征筛选
## 构建筛选特征的算法和参数
sv.methods <- data.frame(tar1.method=c(rep("Enet",11),rep("Stepglm",3),"glmBoost","plsRglm","RF","GBM"),
                         tar1.args=c(seq(0,1,0.1),"backward","forward","both",rep(NA,4))) %>% 
  dplyr::mutate(tar1.method.name=ifelse(is.na(tar1.args),tar1.method,
                                        ifelse(tar1.method=="Enet",str_c(tar1.method,"[alpha=",tar1.args,"]"),
                                               ifelse(tar1.method=="Stepglm",str_c(tar1.method,"[direction=",tar1.args,"]"),tar1.method))),
                tar1.method.name=ifelse(tar1.method.name=="Enet[alpha=1]","Lasso",
                                        ifelse(tar1.method.name=="Enet[alpha=0]","Ridge",tar1.method.name)))
## 筛选特征
preTrain.var <- list() # 用于保存各算法筛选的变量
set.seed(seed = 123456) # 设置建模种子，使得结果可重复
for (x in 1:nrow(sv.methods)){
  method.name <- sv.methods$tar1.method.name[x]
  preTrain.var[[method.name]] = RunML(method = sv.methods$tar1.method[x], # 变量筛选所需要的机器学习方法
                                      method_param = sv.methods$tar1.args[x], # 变量筛选所需要的参数
                                      Train_set = Train_set,         # 训练集有潜在预测价值的变量
                                      Train_label = Train_class,     # 训练集分类标签
                                      mode = "Variable",             # 运行模式，Variable(筛选变量)和Model(获取模型)
                                      classVar = train_outcome)      # 用于训练的分类变量，必须出现在Train_class中
}
preTrain.var <- preTrain.var[sapply(preTrain.var,length)<ncol(Train_set)] # 去除没有筛掉变量的方法
preTrain.var[["simple"]] <- colnames(Train_set) # 新增未经筛选的变量集（以便后续代码撰写）

#### 训练模型 ####
model.train <- list() # 用于保存各模型的所有信息
set.seed(seed = 123456) # 设置建模种子，使得结果可重复
# Train_set_bk = Train_set # RunML有一个函数(plsRglm)无法正常传参，需要对训练集数据进行存档备份
## 算法包括：Lasso、Ridge、Enet、Stepglm、SVM、glmBoost、LDA、plsRglm、RandomForest、GBM、XGBoost、NaiveBayes
## 构建筛选特征的算法和参数
mt.methods <- data.frame(tar2.method=c(rep("Enet",11),rep("Stepglm",3),"SVM","glmBoost","LDA","plsRglm","RF","GBM","XGBoost","NaiveBayes"),
                        tar2.args=c(seq(0,1,0.1),"backward","forward","both",rep(NA,8)))
two.stage.method <- data.frame(tar1=names(preTrain.var)) %>% 
  purrr::pmap_df(~mt.methods %>% dplyr::mutate(tar1.method.name=(...))) %>% 
  dplyr::mutate(tar1.method=str_split(tar1.method.name,"\\[",simplify = T)[,1],
                tar2.method.name=ifelse(is.na(tar2.args),tar2.method,
                                        ifelse(tar2.method=="Enet",str_c(tar2.method,"[alpha=",tar2.args,"]"),
                                               ifelse(tar2.method=="Stepglm",str_c(tar2.method,"[direction=",tar2.args,"]"),tar2.method))),
                tar2.method.name=ifelse(tar2.method.name=="Enet[alpha=1]","Lasso",
                                        ifelse(tar2.method.name=="Enet[alpha=0]","Ridge",tar2.method.name)),
                method.name=ifelse(tar1.method.name=="simple",tar2.method.name,str_c(tar1.method.name," + ",tar2.method.name))) %>% 
  dplyr::filter(tar1.method!=tar2.method)
## 构建各种模型组合
model.train <- list()
for (x in 1:nrow(two.stage.method)){
  cat(x, " Run method :", two.stage.method$method.name[x], "\n")
  method.name = two.stage.method$method.name[x] # 本轮算法名称
  
  Variable = preTrain.var[[two.stage.method$tar1.method.name[x]]] # 根据方法名称的第一个值，调用先前变量筛选的结果
  Train_set_2 = Train_set[, Variable]   # 对训练集取子集，因为有一个算法原作者写的有点问题，无法正常传参
  Train_label = Train_class            # 所以此处需要修改变量名称，以免函数错误调用对象
  model.train[[method.name]] <- RunML(method = two.stage.method$tar2.method[x],     # 根据方法名称第二个值，调用构建的函数分类模型
                                      method_param = two.stage.method$tar2.args[x], # 变量筛选所需要的参数
                                      Train_set = Train_set_2,     # 训练集有潜在预测价值的变量
                                      Train_label = Train_label, # 训练集分类标签
                                      mode = "Model",            # 运行模式，Variable(筛选变量)和Model(获取模型)
                                      classVar = train_outcome)       # 用于训练的分类变量，必须出现在Train_class中
  
  # 如果最终模型纳入的变量数小于预先设定的下限，则认为该算法输出的结果是无意义的
  # if(length(ExtractVar(model.train[[method.name]])) <= min.selected.var) {
  #   model.train[[method.name]] <- NULL
  # }
}
save(model.train, file = "01model.train.RData") 

#### 计算预测和AUC ####
pred.auc.train.res <- lapply(model.train,function(x){CalPredictScore(fit = x,new_data = Train_set,new_class = Train_class,classVar = train_outcome)})
pred.train.res <- lapply(pred.auc.train.res,function(x){x$predict.res})
auc.train.res <- lapply(pred.auc.train.res,function(x){x$auc})

pred.auc.test.res <- lapply(model.train,function(x){CalPredictScore(fit = x,new_data = Test_set,new_class = Test_class,classVar = test_outcome)})
pred.test.res <- lapply(pred.auc.test.res,function(x){x$predict.res})
auc.test.res <- lapply(pred.auc.test.res,function(x){x$auc})

save(pred.train.res,pred.test.res,file = "02predict.score.RData")
save(auc.train.res,auc.test.res,file = "03auc.RData")

#### 计算预测分类(可要可不要) ####
pred.class.train.res <- lapply(model.train, function(x){PredictClass(fit = x, new_data = Train_set)})
pred.class.test.res <- lapply(model.train, function(x){PredictClass(fit = x, new_data = Test_set)})
save(pred.class.train.res,pred.class.test.res,file = "04predict.class.RData")

#### 提取筛选变量 ####
feature.select.res <- lapply(model.train, function(fit){ExtractVar(fit)})
save(feature.select.res, file = "05feature.select.RData")
