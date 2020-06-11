require(data.table)

setwd(".../ESCC_ML-master/example/")
T1 <- read.csv("features_cohort1.csv",header = T)
T2 <- read.csv("features_cohort2.csv",header = T)
T1<-T1[,2:dim(T1)[2]]
T2<-T2[,2:dim(T2)[2]]
dim(T1);dim(T2)
T12 <- rbind(T1,T2);dim(T12)
T12t<-as.data.frame(t(T12))
names(T12t)<-c(1:(dim(T1)[1]+dim(T2)[1]))
str(T12t)

library(sva)
# source("scripts/utils.R");
# source("/scripts/combat.R")
# Author: Jean-Philippe Fortin, fortin946@gmail.com
# This is a modification of the ComBat function code from the sva package that can be found at
# https://bioconductor.org/packages/release/bioc/html/sva.html 

Thead <- read.csv("HEAD.csv",header = T,row.names = 1)
batch = Thead$batch
modcombat = model.matrix(~1, data=Thead)

combat_edata = ComBat(dat=as.matrix(T12t), batch=batch, mod=modcombat, par.prior=FALSE)
combat_edatat<-as.data.frame(t(combat_edata))
str(combat_edatat)
head(combat_edatat)


cohort1<-combat_edatat[1:dim(T1)[1],]
cohort2<-combat_edatat[(dim(T1)[1]+1):(dim(T1)[1]+dim(T2)[1]),]
write.csv(cohort1, quote=FALSE, file=".../ESCC_ML-master/Expected results/harmonized_cohort1.csv", row.names=FALSE)
write.csv(cohort2, quote=FALSE, file=".../ESCC_ML-master/Expected results/harmonized_cohort2.csv", row.names=FALSE)


