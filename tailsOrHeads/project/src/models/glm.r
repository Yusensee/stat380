library(data.table)
train<-fread("./project/volume/data/raw/train_file.csv")
test<-fread("./project/volume/data/raw/test_file.csv")

logr_model<-glm(result ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10,data=train, family=binomial)

summary(logr_model)

saveRDS(logr_model,"./project/volume/models/logr_lm.model")


pred<-predict(logr_model, newdata = test, type="response")

submit<-data.table(id=c(1:100000))

submit$result<-pred

fwrite(submit,"./project/volume/data/processed/submit_logr.csv")