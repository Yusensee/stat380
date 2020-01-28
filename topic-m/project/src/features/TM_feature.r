#TM_feature

library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ClusterR)

train<-fread('./project/volume/data/raw/training_data.csv')
test<-fread('./project/volume/data/raw/test_file.csv')

getEmbeddings<-function(text){
input <- list(
  instances =list( text)
)
res <- POST("https://dsalpha.vmhost.psu.edu/api/use/v1/models/use:predict", body = input,encode = "json", verbose())
emb<-unlist(content(res)$predictions)
emb
}

emb_dt<-NULL
as.data.frame.table(emb_dt)

for (i in 1:length(train$text)){
  emb_dt<-rbind(emb_dt,getEmbeddings(train$text[i]))
  
}
emb_dt<-data.table(emb_dt)

emb_dt_test<-NULL

for (i in 1:length(test$text)){
  emb_dt_test<-rbind(emb_dt_test,getEmbeddings(test$text[i]))
  
}
emb_dt_test<-data.table(emb_dt_test)


fwrite(emb_dt_test,"./project/volume/data/interim/embtest.csv")
fwrite(emb_dt,"./project/volume/data/interim/embtrain.csv")


emb_train<-fread("./project/volume/data/interim/embtrain.csv")
emb_test<-fread("./project/volume/data/interim/embtest.csv")


emb_train <- data.frame(lapply(emb_train, jitter,factor=0.0001))
pca<-prcomp(emb_test,center=TRUE,scale=TRUE)

pca_dt<-data.table(unclass(pca)$x)

tsne<-Rtsne(pca_dt,perplexity=30,max_iter = 3000,check_duplicates = FALSE)

emb_train<-data.table(tsne$Y)
fwrite(emb_train,"./project/volume/data/interim/train_rstne.csv")

emb_test <- data.frame(lapply(emb_test, jitter,factor=0.0001))
pca<-prcomp(emb_test,center=TRUE,scale=TRUE)

pca_dt<-data.table(unclass(pca)$x)

tsne<-Rtsne(pca_dt,perplexity=30,max_iter = 3000,verbose=TRUE,check_duplicates = FALSE)

emb_test<-data.table(tsne$Y)
fwrite(emb_test,"./project/volume/data/interim/test_rstne.csv")