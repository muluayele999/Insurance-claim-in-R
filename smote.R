library(unbalanced)
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv functio
library(DMwR) # smote

setwd("E:/Texas A&M/613 QDA")
ccdata=read.csv("traindatapca.csv",header=TRUE)
#ccdata$C_Claim=ifelse(ccdata$C_Claim=="No",0,1)

table(my.data$C_Claim)
set.seed(1900)
train_smote <- SMOTE(C_Claim ~ ., as.data.frame(ccdata), perc.over = 1000, perc.under=1)
dim(train_smote)
write.csv(train_smote,"trainsmotedata2.csv")
