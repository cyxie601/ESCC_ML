library(psych)

setwd(".../ESCC_ML-master/example")

T1 <- read.csv("features_radiologist1.csv",header = T)
T2 <- read.csv("features_radiologist2.csv",header = T)
T1<-T1[,2:dim(T1)[2]]
T2<-T2[,2:dim(T2)[2]]
dim(T1);dim(T2)
View(T1);View(T2)
x <- dim(T1)[1]
y <- dim(T1)[2]
T12 <- cbind(T1,T2);dim(T12)

icc <- c(1:y)
for(i in 1:y) {icc[i] <- ICC(T12[,c(i,i+y)])$results$ICC[2]}
mean(icc);median(icc)
m <- length(which(icc >= 0.80)) ;m

summary(icc)
dt<-as.data.frame(icc)
dt$FeatureName<-names(T1)
write.csv(dt, file=".../ESCC_ML-master/Expected results/Features_icc.csv")

