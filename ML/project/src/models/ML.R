library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ggplot2)
library(ClusterR)
##########################
# function for picking k #
##########################

getOptK <- function(x){
  x <- abs(x)
  max_delta_k_pos <- which.max(x)
  max_delta_k <- max(na.omit(x))
  n2eval<-(length(x) - max_delta_k_pos) - 2
  for(i in max_delta_k_pos:(max_delta_k_pos+n2eval)){
    if(x[max_delta_k_pos + 1]/max_delta_k < 0.15 & x[max_delta_k_pos + 2]/max_delta_k < 0.15 & x[max_delta_k_pos + 3]/max_delta_k < 0.15){
    }else{
      max_delta_k_pos <- max_delta_k_pos + 1
    }
  }
  max_delta_k_pos
}

data<-fread("./project/volume/data/raw/data.csv")
example_sub<-fread("./project/volume/data/raw/example_sub.csv")
id<-data$id
data$id<-NULL

j_data<-data.frame(lapply(data, jitter,factor=0.01))

pca<-prcomp(j_data)

pca_dt<-data.table(unclass(pca)$x)

pca_dt$id<-id


tsne<-Rtsne(pca_dt,pca = F)


tsne_dt<-data.table(tsne$Y)

tsne_dt$party<-id

k_bic<-Optimal_Clusters_GMM(tsne_dt[,.(V1,V2)],max_clusters = 10,criterion = "BIC")

delta_k<-c(NA,k_bic[-1] - k_bic[-length(k_bic)])

del_k_tab<-data.table(delta_k=delta_k,k=1:length(delta_k))

opt_k<-getOptK(delta_k)
gmm_data<-GMM(tsne_dt[,.(V1,V2)],opt_k)#k=3


l_clust<-gmm_data$Log_likelihood^10

l_clust<-data.table(l_clust)

net_lh<-apply(l_clust,1,FUN=function(x){sum(1/x)})

cluster_prob<-1/l_clust/net_lh

cluster_prob$id<-id
example_sub$species1<-cluster_prob$V2
example_sub$species2<-cluster_prob$V3
example_sub$species3<-cluster_prob$V1
example_sub$species4<-0
example_sub$species5<-0
example_sub$species6<-0
example_sub$species7<-0
example_sub$species8<-0
example_sub$species9<-0
example_sub$species10<-0
fwrite(example_sub,"./project/volume/data/processed/submit_3.csv")

fwrite(emb_dt,"./project/volume/data/processed/embtrain.csv")