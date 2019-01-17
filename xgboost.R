install.packages("randomForest")
install.packages("unbalanced")
install.packages("DMwR")
install.packages("penalisedSVM")
install.packages("caret")
install.packages("xgboost")
install.packages("reshape")
install.packages("pROC")
library(penalisedSVM)
library(DMwR)
library(PCAmixdata)
library(ISLR)
library(glmnet)
library(boot)
library(MASS)
library(class)
library(car)
library(unbalanced)
library(e1071)
library(randomForest)
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret)
library(DMwR) # smote
library(xgboost)
library(Matrix)
library(reshape) #melt
library(pROC) # AUC
setwd("E:/Texas A&M/613 QDA")
ccdata=read.csv("allpcadata.csv",header=TRUE)
ccdata$C_Claim=ifelse(ccdata$C_Claim=="No",0,1)

table(my.data$C_Claim)
set.seed(1900)
inTrain <- createDataPartition(y = ccdata$C_Claim, p = .8, list = F)
train <- ccdata[inTrain,]
testcv <- ccdata[-inTrain,]
inTest <- createDataPartition(y = testcv$C_Claim, p = .5, list = F)
test <- testcv[inTest,]
cv <- testcv[-inTest,]
train$C_Claim <- as.factor(train$C_Claim)
rm(inTrain, inTest, testcv)

i <- grep("C_Claim", colnames(train)) # Get index Class column
train_smote <- SMOTE(C_Claim ~ ., as.data.frame(train), perc.over = 2000, perc.under=100)

table(train_smote$Class)
summary(train_smote)

train$C_Claim <- as.numeric(levels(train$C_Claim))[train$C_Claim]
train_smote$C_Claim <- as.numeric(levels(train_smote$C_Claim))[train_smote$C_Claim]

#As Matrix
train <- Matrix(as.matrix(train),sparse = TRUE)
train_smote <- Matrix(as.matrix(train_smote), sparse = TRUE)
test <- Matrix(as.matrix(test), sparse = TRUE)
cv <- Matrix(as.matrix(cv), sparse = TRUE)

# Create XGB Matrices
train_xgb <- xgb.DMatrix(data = train[,-i], label = train[,i])
train_smote_xgb <- xgb.DMatrix(data = train_smote[,-i], label = train_smote[,i])
test_xgb <- xgb.DMatrix(data = test[,-i], label = test[,i])
cv_xgb <- xgb.DMatrix(data = cv[,-i], label = cv[,i])

# Watchlist
watchlist <- list(train  = train_xgb, cv = cv_xgb)
parameters <- list(
  # General Parameters
  booster            = "gbtree",          
  silent             = 0,                 
  # Booster Parameters
  eta                = 3,               
  gamma              = 0,                 
  max_depth          = 7,                 
  min_child_weight   = 1,                 
  subsample          = 1,                 
  colsample_bytree   = 1,                 
  colsample_bylevel  = 1,                 
  lambda             = 1,                 
  alpha              = 0,   
  scale_pos_weight   = 136,
  max_delta_step     = 2,
  # Task Parameters
  objective          = "binary:logistic",
  eval_metric        = "error@0.95",
  seed               = 1900               
)
#Claim=ifelse(test[,i]>0,1,0)
# Original
xgb.model <- xgb.train(parameters, train_xgb, nrounds = 100, watchlist)

# Smote
xgb_smote.model <- xgb.train(parameters, train_smote_xgb, nrounds = 50, watchlist)

# Threshold
q <-  0.95

# Original
xgb.predict <- predict(xgb.model, test)
xgb.predictboolean <- ifelse(xgb.predict > q,1,0)  
roc <- roc(test[,i], predict(xgb.model, test, type = "prob"))
xgb.cm <- confusionMatrix(xgb.predictboolean, test[,i])
xgb.cm$table
print(paste("AUC of XGBoost is:",roc$auc))
print(paste("F1 of XGBoost is:", xgb.cm$byClass["F1"]))
xgb.cm$byClass

# SMOTE
roc_smote <- roc(test[,i], predict(xgb_smote.model, test, type = "prob"))
xgb_smote.predict <- predict(xgb_smote.model, test)
xgb_smote.predictboolean <- ifelse(xgb_smote.predict >= q,1,0)  
xgb_smote.cm <- confusionMatrix(xgb_smote.predictboolean, test[,i])
xgb_smote.cm$table
