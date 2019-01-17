install.packages("e1071")
install.packages("PCAmixdata")
install.packages("car")
install.packages("ISLR")
library(xlsx)
library(PCAmixdata)
library(ISLR)
library(glmnet)
library(boot)
library(MASS)
library(class)
library(tree)
library(car)
library(e1071)
library(party)
# Load The Data
dir()
setwd("E:/Texas A&M/613 QDA")
my.data=read.csv("train_set.csv",header=TRUE)
my.data2 = my.data
# Remove Columns 1:8
my.data[ ,(1:8) ]=NULL 
my.data2[,(1:2)] = NULL
# Create C_Claim (Categorical variable)
C_Claim=ifelse(my.data$Claim_Amount >0,"Yes","No")
my.data=cbind(my.data,C_Claim)

# Remove Cat2,4,5,7 dut to lot of "?"
#my.data[ ,c(2,4,5,7)]=NULL
# Replace all ? with NA
#my.data[my.data == "?"] <- NA
#my.data2[my.data2 == "?"] <- NA

sum(is.na(my.data))


# Remove Claim-Ammount Column
my.data[ ,27]=NULL
my.data2[,33]=NULL
#dim(my.data)
#sum(is.na(my.data))
#sum(my.data$C_Claim=="Yes")     # 717

#sum(my.data2$C_Claim2=="Yes")     # 717

# Apply PCA
npca.data=my.data[,c(1:13,22,27)]
pca.data=my.data[,-c(1:13,22,27)]

apply(pca.data, 2, mean)
apply(pca.data, 2, var)

##apply PCA
pr.out=prcomp(pca.data, scale=TRUE)
my.data2$Vehicle <- as.character(my.data2$Vehicle)#remove some categorical variable
my.data2$Calendar_Year <- as.character(my.data2$Calendar_Year)
my.data2$Model_Year <- as.character(my.data2$Model_Year)
my.data2$Blind_Make<-as.character(my.data2$Blind_Make)
my.data2[,c(1:4,7:19,28)] = as.character(my.data2[,c(1:4,7:19,28)])
#qualitative PCA
quali.pca <- PCAmix(X.quanti = NULL, X.quali = my.data2[,c(1:4,7:19,28)],ndim=11,rename.level = TRUE )
plot(quali.pca$eig[,2], xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
rotated<-PCArot(quali.pca,dim = 11,itermax = 100)
scores<-quali.pca$scores #final rotated PCA
dim(scores)

names(pr.out)
pr.out$center     #means of the X variables
pr.out$scale      #stds of the X variables
pr.out$rotation   #loading matrix
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", type='b')
dim(pr.out$x)                 #x: score vectors (Zmatrix)
my.data.pca=cbind(scores,pr.out$x[,1:10],C_Claim)
#calling the pca applied data set
write.csv(my.data.pca[1:100000,],"traindatapca.csv")
write.csv(my.data.pca[100001:150000,],"testdatapca.csv")

my.data.pca=read.csv("datapca.csv",header=TRUE)


#dividing the data set into test and train sets using the ratio of 0.9
set.seed(100)
samples=sample(nrow(my.data.pca), 0.8*nrow(my.data.pca))
train = (my.data.pca[samples, ])
test = my.data.pca[-samples,]
#creating the conditional inference decision tree
z<-ctree(C_Claim~.-(C_Claim),data=train)
z
plot(z)
#predicting the results on test data
pred<-predict(z,newdata=test)
#creating confusion matrix
table(pred,test$C_Claim)
