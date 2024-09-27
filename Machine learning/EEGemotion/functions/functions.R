#### 定义各个方法的函数 ####
RunEnet <- function(Train_set, Train_label, mode, classVar, alpha, nfolds=10){
  cv.fit = cv.glmnet(x = Train_set,
                     y = Train_label[[classVar]],
                     family = "binomial", alpha = alpha, nfolds = nfolds)
  fit = glmnet(x = Train_set,
               y = Train_label[[classVar]],
               family = "binomial", alpha = alpha, lambda = cv.fit$lambda.min)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunLasso <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  RunEnet(Train_set, Train_label, mode, classVar, alpha = 1, nfolds)
}

RunRidge <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  RunEnet(Train_set, Train_label, mode, classVar, alpha = 0, nfolds)
}

RunStepglm <- function(Train_set, Train_label, mode, classVar, direction){
  fit <- step(glm(formula = Train_label[[classVar]] ~ .,
                  family = "binomial", 
                  data = as.data.frame(Train_set)),
              direction = direction, trace = 0)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunSVM <- function(Train_set, Train_label, mode, classVar){
  data <- as.data.frame(Train_set)
  data[[classVar]] <- as.factor(Train_label[[classVar]])
  fit = svm(formula = eval(parse(text = paste(classVar, "~."))),
            data= data, probability = T)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunLDA <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  data <- as.data.frame(Train_set)
  data[[classVar]] <- as.factor(Train_label[[classVar]])
  fit = train(eval(parse(text = paste(classVar, "~."))), 
              data = data, 
              method="lda",
              trControl = trainControl(method = "cv",number = nfolds))
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunglmBoost <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  data <- cbind(Train_set, Train_label[classVar])
  data[[classVar]] <- as.factor(data[[classVar]])
  fit <- glmboost(eval(parse(text = paste(classVar, "~."))),
                  data = data,
                  family = Binomial())
  
  cvm <- cvrisk(fit, papply = lapply,
                folds = cv(model.weights(fit), type = "kfold", B=nfolds))
  fit <- glmboost(eval(parse(text = paste(classVar, "~."))),
                  data = data,
                  family = Binomial(), 
                  control = boost_control(mstop = max(mstop(cvm), 100)))
  
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunplsRglm <- function(Train_set, Train_label, mode, classVar){
  # cv.plsRglm.res = cv.plsRglm(formula = Train_label[[classVar]] ~ ., 
  #                             data = as.data.frame(Train_set),
  #                             nt=10, verbose = FALSE)
  fit <- plsRglm(Train_label[[classVar]], 
                 as.data.frame(Train_set), 
                 modele = "pls-glm-logistic",
                 verbose = F, sparse = T)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunRF <- function(Train_set, Train_label, mode, classVar){
  rf_nodesize = 5 # may modify
  Train_label[[classVar]] <- as.factor(Train_label[[classVar]])
  fit <- rfsrc(formula = formula(paste0(classVar, "~.")),
               data = cbind(Train_set, Train_label[classVar]),
               ntree = 1000, nodesize = rf_nodesize,
               importance = T,
               proximity = T,
               forest = T)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunGBM <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  fit <- gbm(formula = Train_label[[classVar]] ~ .,
             data = as.data.frame(Train_set),
             distribution = 'bernoulli',
             n.trees = 1000,
             interaction.depth = 1,
             n.minobsinnode = 10,
             shrinkage = 0.1,
             cv.folds = nfolds,n.cores = 6)
  best <- which.min(fit$cv.error)
  fit <- gbm(formula = Train_label[[classVar]] ~ .,
             data = as.data.frame(Train_set),
             distribution = 'bernoulli',
             n.trees = best,
             interaction.depth = 1,
             n.minobsinnode = 10,
             shrinkage = 0.1, n.cores = 6)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunXGBoost <- function(Train_set, Train_label, mode, classVar, nfolds=10){
  indexes = createFolds(Train_label[[classVar]], k = nfolds, list=T)
  CV <- unlist(lapply(indexes, function(pt){
    dtrain = xgb.DMatrix(data = Train_set[-pt, ], 
                         label = Train_label[-pt, ])
    dtest = xgb.DMatrix(data = Train_set[pt, ], 
                        label = Train_label[pt, ])
    watchlist <- list(train=dtrain, test=dtest)
    
    bst <- xgb.train(data=dtrain, 
                     nthread = 2, nrounds=10, 
                     watchlist=watchlist, 
                     objective = "binary:logistic", verbose = F)
    which.min(bst$evaluation_log$test_logloss)
  }))
  
  nround <- as.numeric(names(which.max(table(CV))))
  fit <- xgboost(data = Train_set, 
                 label = Train_label[[classVar]], 
                 nthread = 5, nrounds = nround, 
                 objective = "binary:logistic", verbose = F)
  fit$subFeature = colnames(Train_set)
  
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

RunNaiveBayes <- function(Train_set, Train_label, mode, classVar){
  data <- cbind(Train_set, Train_label[classVar])
  data[[classVar]] <- as.factor(data[[classVar]])
  fit <- naiveBayes(eval(parse(text = paste(classVar, "~."))), 
                    data = data)
  fit$subFeature = colnames(Train_set)
  if (mode == "Model") return(fit)
  if (mode == "Variable") return(ExtractVar(fit))
}

#### 提取features ####
ExtractVar <- function(fit){
  Feature <- suppressMessages({
    switch(
      EXPR = class(fit)[1],
      "lognet" = rownames(coef(fit))[which(coef(fit)[, 1]!=0)], # 本身没有筛选变量的功能，但是可以舍去模型中系数为0的变量
      "glm" = names(coef(fit)), # 逐步回归可以对变量进行筛选
      "svm.formula" = fit$subFeature, # SVM对变量没有筛选功能，所以默认使用所有变量
      "train" = fit$coefnames, # LDA不能筛选变量，所以默认使用所有变量
      "glmboost" = names(coef(fit)[abs(coef(fit))>0]), # glmboost同样不具备筛选变量的功能，因此舍去模型中系数为0的变量
      "plsRglmmodel" = rownames(fit$Coeffs)[fit$Coeffs!=0], # plsRglmmodel同样不具备筛选变量的功能，因此舍去模型中系数为0的变量
      "rfsrc" = var.select(fit, verbose = F)$topvars, # rfsrc可以对变量进行筛选
      "gbm" = rownames(summary.gbm(fit, plotit = F))[summary.gbm(fit, plotit = F)$rel.inf>0], # gbm通过舍去重要性为0的变量来进行变量筛选
      "xgb.Booster" = fit$subFeature, # xgboost没有筛选变量的能力， 默认使用所有变量
      "naiveBayes" = fit$subFeature # naiveBayes没有筛选变量的能力，默认使用所有变量
      # "drf" = fit$subFeature # drf自带的变量系数提取函数会输出NA，因此默认使用所有变量
    )
  })
  
  Feature <- setdiff(Feature, c("(Intercept)", "Intercept"))
  return(Feature)
}

#### 运行函数 ####
RunML <- function(method, method_param=NA, Train_set, Train_label, mode = "Model", classVar){
  
  method_name=method
  
  if(!is.na(method_param)){
    method_param = switch(
      EXPR = method_name,
      "Enet" = list("alpha" = as.numeric(method_param)),
      "Stepglm" = list("direction" = method_param),
      NULL
    )
  }
  
  message("Run ", method_name, " algorithm for ", mode, "; ",
          method_param, ";",
          " using ", ncol(Train_set), " Variables")
  
  args = list("Train_set" = Train_set,
              "Train_label" = Train_label,
              "mode" = mode,
              "classVar" = classVar)
  
  if(!is.na(method_param)){
    args = c(args, method_param)
  }
  
  obj <- do.call(what = paste0("Run", method_name),
                 args = args) 
  
  if(mode == "Variable"){
    message(length(obj), " Variables retained;\n")
  }else{message("\n")}
  return(obj)
}

#### 计算预测值和AUC ####
CalPredictScore <- function(fit, new_data, new_class, classVar, type = "lp"){
  new_data <- new_data[, fit$subFeature]
  RS <- suppressMessages({
    switch(
      EXPR = class(fit)[1],
      "lognet"      = predict(fit, type = 'response', as.matrix(new_data)), # response
      "glm"         = predict(fit, type = 'response', as.data.frame(new_data)), # response
      "svm.formula" = predict(fit, as.data.frame(new_data), probability = T), # 
      "train"       = predict(fit, new_data, type = "prob")[[2]],
      "glmboost"    = predict(fit, type = "response", as.data.frame(new_data)), # response
      "plsRglmmodel" = predict(fit, type = "response", as.data.frame(new_data)), # response
      "rfsrc"        = predict(fit, as.data.frame(new_data))$predicted[, "1"],
      "gbm"          = predict(fit, type = 'response', as.data.frame(new_data)), # response
      "xgb.Booster" = predict(fit, as.matrix(new_data)),
      "naiveBayes" = predict(object = fit, type = "raw", newdata = new_data)[, "1"]
      # "drf" = predict(fit, functional = "mean", as.matrix(new_data))$mean
    )
  })
  RS = as.numeric(as.vector(RS))
  names(RS) = rownames(new_data)
  new_class <- new_class %>% dplyr::mutate(pred=RS)
  fml <- as.formula(paste0(classVar,"~pred"))
  auc.res <- as.numeric(auc(suppressMessages(roc(fml, data = new_class, smooth=F))))
  res.list <- list(predict.res=RS,auc=auc.res)
  return(res.list)
}

#### 预测分类情况 ####
PredictClass <- function(fit, new_data){
  new_data <- new_data[, fit$subFeature]
  label <- suppressMessages({
    switch(
      EXPR = class(fit)[1],
      "lognet"      = predict(fit, type = 'class', as.matrix(new_data)),
      "glm"         = ifelse(test = predict(fit, type = 'response', as.data.frame(new_data))>0.5, 
                             yes = "1", no = "0"), # glm不返回预测的类，将概率>0.5的作为1类
      "svm.formula" = predict(fit, as.data.frame(new_data), decision.values = T), # 
      "train"       = predict(fit, new_data, type = "raw"),
      "glmboost"    = predict(fit, type = "class", as.data.frame(new_data)), 
      "plsRglmmodel" = ifelse(test = predict(fit, type = 'response', as.data.frame(new_data))>0.5, 
                              yes = "1", no = "0"), # plsRglm不允许使用因子变量作为因变量，因而predict即使type设为class也无法正常运作
      "rfsrc"        = predict(fit, as.data.frame(new_data))$class,
      "gbm"          = ifelse(test = predict(fit, type = 'response', as.data.frame(new_data))>0.5,
                              yes = "1", no = "0"), # gbm未设置预测类别，设置大于0.5为1
      "xgb.Booster" = ifelse(test = predict(fit, as.matrix(new_data))>0.5,
                             yes = "1", no = "0"), # xgboost 未提供预测类别，设置大于0.5为1
      "naiveBayes" = predict(object = fit, type = "class", newdata = new_data)
      # "drf" = predict(fit, functional = "mean", as.matrix(new_data))$mean
    )
  })
  label = as.character(as.vector(label))
  names(label) = rownames(new_data)
  return(label)
}
