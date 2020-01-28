library(data.table)

train<-fread("./project/volume/data/raw/Stat_380_train.csv")
test<-fread("./project/volume/data/raw/Stat_380_test.csv")

train[,"age":= YrSold-YearBuilt]
test[,"age":= YrSold-YearBuilt]

input <- train[,c("LotArea","OverallQual","OverallCond","FullBath","HalfBath","TotRmsAbvGrd","TotalBsmtSF","BedroomAbvGr","GrLivArea","age","SalePrice","YrSold")]

model<-lm(SalePrice~LotArea+OverallQual+OverallCond+FullBath+HalfBath+TotRmsAbvGrd+TotalBsmtSF+age, data = input)

saveRDS(model,"./project/volume/models/lm.model")

test$SalePrice<-predict(model,newdata = test[,.(LotArea,OverallQual,OverallCond,FullBath,HalfBath,TotRmsAbvGrd,TotalBsmtSF,age)])

submit<-test[,.(Id,SalePrice)][order(Id)]

fwrite(submit,"./project/volume/data/processed/submit_lm.csv")