#run feature frist
library(keras)
library(tidyverse)
library(xgboost)
library(caret)
library(Rtsne)
library(Metrics)
library(data.table)

train<-fread('./project/volume/data/raw/training_data.csv')
data<-fread('./project/volume/data/raw/training_data.csv')
test<-fread('./project/volume/data/raw/test_file.csv')
#make labels
labels <- data$id
labels <- as.data.frame(labels)
labels$SUM <-train[, 3] * 1+train[, 4] * 2+train[, 5] * 3+train[, 6] * 4+train[, 7] * 5+train[, 8] * 6+train[, 9] * 7+train[, 10] * 8+train[, 11] * 9+train[, 12] * 10 -1

# this part is for version1 use just use the emb and then xgboost
xgb_train <- xgb.DMatrix(data = as.matrix(emb_train), label = as.matrix(labels$SUM))

params <- list(booster = "gbtree", 
	  objective = "multi:softprob", 
	  num_class = 10, 
	  max_depth = 9,
      eta = 0.06947056,
      gamma = 0.07571645, 
      subsample = 0.6862556,
      colsample_bytree = 0.5751902, 
      min_child_weight = 2,
      max_delta_step = 2,#nrounds139
      eval_metric = "mlogloss")
	
xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 139, nfold = 5, showsd = TRUE, stratified = TRUE,  
                print_every_n = 10, early_stop_round = 20, nthread=6,maximize = TRUE, verbose = T,prediction = TRUE)

xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = as.matrix(labels$SUM + 1))

xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 139)#change nround to same nrounds above

xgb_val_preds <- as.data.frame(predict(xgb_model, newdata = as.matrix(emb_test)))

trans <- as.data.frame(split(xgb_val_preds, 1:10))

colnames(trans) <- c("subredditcars", "subredditCooking", "subredditMachineLearning", "subredditMagicTCG", "subredditpolitics",
                     "subredditReal_Estate", "subredditscience", "subredditStockMarket", "subreddittravel", "subredditvideogames")

trans$id <- test$id

fwrite(trans,"./project/volume/data/processed/submit_final_V1.csv")

# this part is for version2, combine the emb and the data of rstne together then do xgboost
emb_train<-fread("./project/volume/data/interim/train_model.csv")
emb_train<-emb_train[,c(1:514)]
emb_test<-fread("./project/volume/data/interim/test_model.csv")

xgb_train <- xgb.DMatrix(data = as.matrix(emb_train), label = as.matrix(labels$SUM))

params <- list(booster = "gbtree", 
	  objective = "multi:softprob", 
	  num_class = 10, 
	  max_depth = 6,
      eta = 0.05319178,
      gamma = 0.1831914, 
      subsample = 0.6073907,
      colsample_bytree = 0.782241, 
      min_child_weight = 1,
      max_delta_step = 2,
      eval_metric = "mlogloss")

xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 179, nfold = 5, showsd = TRUE, stratified = TRUE,  
                print_every_n = 10, early_stop_round = 20, nthread=6,maximize = TRUE, verbose = T,prediction = TRUE)

xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = as.matrix(labels$SUM + 1))

xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 179)#change nrounds to same nrounds above

xgb_val_preds <- as.data.frame(predict(xgb_model, newdata = as.matrix(emb_test)))
#or 
#xgb_val_preds <- as.data.frame(predict(xgb_model, newdata = as.matrix(bert_test)))

trans <- as.data.frame(split(xgb_val_preds, 1:10))

colnames(trans) <- c("subredditcars", "subredditCooking", "subredditMachineLearning", "subredditMagicTCG", "subredditpolitics",
                     "subredditReal_Estate", "subredditscience", "subredditStockMarket", "subreddittravel", "subredditvideogames")

trans$id <- test$id

fwrite(trans,"./project/volume/data/processed/submit_final_v2.csv")

#############################################
# this part is the iterations to find the best param for xgboost
# try booster gblinear not good
# gbtree good
# dart not good
# try tree method approx bot good
# read the best_param and nround after iteration and set up param manually 
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
    param <- list(booster = "gbtree",
    	  objective = "multi:softprob",
          eval_metric = "mlogloss",
          num_class = 10,
          max_depth = sample(6:10, 1),
          eta = runif(1, .01, .3),
          gamma = runif(1, 0.0, 0.2), 
          subsample = runif(1, .6, .9),
          colsample_bytree = runif(1, .5, .8), 
          min_child_weight = sample(1:40, 1),
          max_delta_step = sample(1:10, 1)
          )
    cv.nround = 1000
    cv.nfold = 5
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv <- xgb.cv(data=xgb_train, params = param, nthread=6, 
                    nfold=cv.nfold, nrounds=cv.nround,
                    verbose = T, early_stop_round=8, maximize=FALSE)

    min_logloss = min(mdcv$evaluation_log[, test_mlogloss_mean])
    min_logloss_index = which.min(mdcv$evaluation_log[, test_mlogloss_mean])

    if (min_logloss < best_logloss) {
        best_logloss = min_logloss
        best_logloss_index = min_logloss_index
        best_seednumber = seed.number
        best_param = param
    }
}


nround = best_logloss_index
set.seed(best_seednumber)
xgbcv<- xgb.train(data=xgb_train, params=best_param, nrounds=nround, nthread=6,showsd = TRUE, stratified = TRUE,  
                print_every_n = 10, early_stop_round = 20,maximize = TRUE,verbose = T,prediction = TRUE)

###############VERY IMPROTANT##############
###CLEAR THE LOACAL AFTER ITERATIONS!!!!###
###########################################

#reference
#https://rpubs.com/zxs107020/368478
#https://stackoverflow.com/questions/35050846/xgboost-in-r-how-does-xgb-cv-pass-the-optimal-parameters-into-xgb-train


###########################################
#code below is some other pramas and attempts

#### below is the best params for jitter pca rstne 
#params <- list(booster = "gbtree", 
	  #objective = "multi:softprob", 
	  #num_class = 10, 
	  #max_depth = 10,
      #eta = 0.09158767,
      #gamma = 0.04581406, 
      #subsample = 0.6133814,
      #colsample_bytree = 0.6075862, 
      #min_child_weight = 3,
      #max_delta_step = 1,#nrounds217
      #eval_metric = "mlogloss")
	
#xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 217, nfold = 5, showsd = TRUE, stratified = TRUE,  
                #print_every_n = 10, early_stop_round = 20, nthread=6,maximize = TRUE, verbose = T,prediction = TRUE)
#########################################
#### below is the best params for bert in 100 iterations 
#params <- list(booster = "gbtree", 
	  #tree_method ="approx",
	  #objective = "multi:softprob", 
	  #num_class = 10, 
	 # max_depth = 9,
      #eta = 0.1554213,
      #gamma = 0.01504133, 
      #subsample = 0.726527,
      #colsample_bytree = 0.7144218, 
      #min_child_weight = 1,
      #max_delta_step = 7,
      #eval_metric = "mlogloss")

#xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 108, nfold = 5, showsd = TRUE, stratified = TRUE,  
                #print_every_n = 10, early_stop_round = 20, nthread=6,maximize = TRUE, verbose = T,prediction = TRUE)

#emb_train<-fread("./project/volume/data/interim/embtrain.csv")
#emb_test<-fread("./project/volume/data/interim/embtest.csv")
#use bert
#bert_train<-fread("./project/volume/data/interim/train_bert.csv")
#bert_test<-fread("./project/volume/data/interim/test_bert2.csv")

############
#emb_train<-fread("./project/volume/data/interim/embtrain.csv")
#emb_test<-fread("./project/volume/data/interim/embtest.csv")
#use bert
#bert_train<-fread("./project/volume/data/interim/train_bert.csv")
#bert_test<-fread("./project/volume/data/interim/test_bert2.csv")

#emb_train <- data.frame(lapply(emb_train, jitter,factor=0.01))
#emb_train <- Rtsne(emb_train,check_duplicates = FALSE)
#emb_train <- data.table(emb_train$Y)

#emb_test <- data.frame(lapply(emb_test, jitter,factor=0.0001))
#emb_test <- Rtsne(emb_test,check_duplicates = FALSE)
#emb_test <- data.table(emb_test$Y)

############## jitter pca and rtsne
#emb_train <- data.frame(lapply(emb_train, jitter,factor=0.0001))
#pca<-prcomp(emb_test,center=TRUE,scale=TRUE)

#pca_dt<-data.table(unclass(pca)$x)

#tsne<-Rtsne(pca_dt,perplexity=30,max_iter = 3000,check_duplicates = FALSE)

#emb_train<-data.table(tsne$Y)
#emb_train<-fread("./project/volume/data/interim/train_rstne.csv")

#emb_test <- data.frame(lapply(emb_test, jitter,factor=0.0001))
#pca<-prcomp(emb_test,center=TRUE,scale=TRUE)

#pca_dt<-data.table(unclass(pca)$x)

#tsne<-Rtsne(pca_dt,perplexity=30,max_iter = 3000,verbose=TRUE,check_duplicates = FALSE)

#emb_test<-data.table(tsne$Y)
#emb_test<-fread("./project/volume/data/interim/test_rstne.csv")

