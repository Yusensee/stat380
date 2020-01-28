library(data.table)
library(caret)
library(Metrics)
library(lubridate)
#the 1.0 verstion use code on github and work bad
#the 1.1 version try lots of attributes like colors,mana_cost,type,rarity and not work well
#but find one important attribute which is rarity_mythic
#the 1.2 version based on the price history,ADD 7day-price-change predict future price
example_sub<-fread("./project/volume/data/raw/example_submission.csv")
start_test<-fread("./project/volume/data/raw/start_test.csv")
start_train<-fread("./project/volume/data/raw/start_train.csv")
set_tab<-fread("./project/volume/data/raw/set_tab.csv")
card_tab<-fread("./project/volume/data/raw/card_tab.csv")
price_tab<-fread("./project/volume/data/raw/price_tab.csv") 
#load form external, data from https://www.mtggoldfish.com/index/ELD#paper and https://www.mtgstocks.com/sets/331 on Oct/11/2019
test_price<-fread("./project/volume/data/external/7daypirce_test.csv")

keepset<- c('set','id')
card_keep<-card_tab[,..keepset]

keepset<- c('set','release_date')
set_keep<-set_tab[,..keepset]

info<-merge(card_keep,set_keep,by.x="set",by.y = "set")
info$release_date <- ymd(info$release_date)
info[,release_date := release_date + days(7)]

merge_trade<-merge(price_tab,info,by.x = "id",by.y="id")
merge_trade$date <- ymd(merge_trade$date)
setnames(merge_trade, old=c("release_date"), new=c("date7"))
# get the price of 7 day after released
trade7<-merge_trade[date == date7]

trade_train<-merge(start_train,trade7,by.x = "id",by.y = "id")


trade_test<-merge(start_test,test_price,by.x = "id",by.y = "id")

setnames(trade_test, old=c("Price"), new=c("price"))

keepset<-c("id","current_price","future_price","price")
trade_train<-trade_train[,..keepset]
trade_test$future_price<-0
trade_test<-trade_test[,..keepset]




setnames(trade_train, old=c("price"), new=c("price7day"))
setnames(trade_test, old=c("price"), new=c("price7day"))

trade_train[,"change_price":= future_price-current_price]

trade_train[,"change_7day":= price7day-current_price]
trade_test[,"change_7day":= price7day-current_price]

####just use a lm
#lm<-line_model<-lm(future_price ~ current_price + change_7day,data=trade_train)


#trade_test$future_price<-predict(line_model, newdata = trade_test)
#example_sub$future_price<-trade_test$future_price
#fwrite(example_sub,"./project/volume/data/processed/submit_1.2.csv")



##########try other model 1.2.1 not good
#line_model2<-lm( change_price~ change_7day,data=trade_train)
#trade_test$change_price<-predict(line_model2, newdata = trade_test)
#trade_test[,future_price:=current_price+change_price]
##########some vaule become negative number, which is can't happen in realworld

#add rarity
keepset<- c('rarity','id')
card_keep<-card_tab[,..keepset]

trade_train<-merge(trade_train,card_keep,by.x = "id",by.y = "id")

trade_test<-merge(trade_test,card_keep,by.x = "id",by.y = "id")
#take Mythic with other group
trade_train_mythic<-trade_train[rarity=="Mythic"]
trade_train<-trade_train[rarity!="Mythic"]

trade_test_mythic<-trade_test[rarity=="Mythic"]
trade_test<-trade_test[rarity!="Mythic"]

#fit a model with current_price and change_7day
lm_NOmythic<-lm(future_price ~ current_price + change_7day,data=trade_train)
lm_mythic<-lm(future_price ~ current_price + change_7day,data=trade_train_mythic)

saveRDS(lm_NOmythic,"./project/volume/models/lm_NOmythic.model")
saveRDS(lm_mythic,"./project/volume/models/lm_mythic.model")

trade_test$future_price<-predict(lm_NOmythic, newdata = trade_test)
trade_test_mythic$future_price<-predict(lm_mythic, newdata = trade_test_mythic)

trade_test2<-rbind(trade_test,trade_test_mythic)
example_sub<-merge(example_sub,trade_test2,by.x = "id",by.y = "id")

keepset<-c("id","future_price.y")
example_sub<-example_sub[,..keepset]
setnames(example_sub, old=c("future_price.y"), new=c("future_price"))

fwrite(example_sub,"./project/volume/data/processed/submit_final.csv")