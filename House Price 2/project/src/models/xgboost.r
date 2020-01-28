library(data.table)
library(caret)
library(Metrics)
library(glmnet)
library(xgboost)

train<-fread("./project/volume/data/raw/Stat_380_train.csv")
test<-fread("./project/volume/data/raw/Stat_380_test.csv")
example_sub <- fread("./project/volume/data/raw/Stat_380_sample_submission.csv")


train[is.na(train)]<-0
test[is.na(test)]<- 0

sub_train<-train[,.(Heating,BldgType,SalePrice)]
sub_test<-test[,.(Heating,BldgType)]

sub_test$SalePrice<-1
sub_train_y<-sub_train$SalePrice
dummies <- dummyVars(SalePrice ~ ., data = sub_train)
sub_train<-predict(dummies, newdata = sub_train)
sub_test<-predict(dummies, newdata = sub_test)

sub_train<-data.table(sub_train)
sub_test<-data.table(sub_test)

sub_test[, ID := .I]
sub_train[, ID := .I]

train[,"age":= YrSold-YearBuilt]
test[,"age":= YrSold-YearBuilt]

merge_train <- merge(train,sub_train,by.x = 'Id', by.y='ID')
merge_test <- merge(test,sub_test,by.x = 'Id', by.y='ID')

drop <- c('Heating','BldgType','CentralAir')
train1 <- merge_train[, !drop, with = FALSE]
test1 <- merge_test[, !drop, with = FALSE]

x.train <- as.matrix(train1[, c(2:14,16:25)])
y.train <- as.matrix(train1$SalePrice)
x.test <- as.matrix(test1[,c(2:24)])

dtrain <- xgb.DMatrix(x.train,label=y.train,missing=NA)
dtest <- xgb.DMatrix(x.test,missing=NA)

param <- list(  objective           = "reg:linear",
                gamma               = 0 ,
                booster             = "gbtree",
                eval_metric         = "rmse",
                eta                 = 0.02,
                max_depth           = 5,
                subsample           = 0.9,
                colsample_bytree    = 0.9,
                tree_method = 'hist'
)

XGBm<-xgb.cv(params=param,nfold=5,nrounds=700,missing=NA,data=dtrain,print_every_n=1) 

XGBm<-xgb.train(params=param,nrounds=1000,missing=NA,data=dtrain,print_every_n=1)

#XGBm<-xgb.train( params=param,nrounds=500,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)

pred<-predict(XGBm, newdata = dtest)
example_sub$SalePrice <- pred
fwrite(example_sub,"./project/volume/data/processed/submit_final.csv")
