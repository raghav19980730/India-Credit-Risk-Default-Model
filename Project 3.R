#Setting up working directories and loading libraries
setwd("C:/Users/ragha/Downloads")
library(readxl)
library(tidyverse)
library(ROCR)
library(car)
library(randomForest)
library(rpart)
library(caTools)
library(rpart.plot)
library(caret)
library(ineq)
library(gridExtra)

#Importing the dataset

bank <- read_excel("Thera Bank_Personal_Loan_Modelling-dataset-1.xlsx", sheet = 2)
bank <- as.data.frame(bank[,-c(1,5)])
colnames(bank) <- c("Age","Experience","Income","Family.Members","CCAvg","Education",
                     "Mortgage","Personal.Loan", "Security.Account", "CD.Account", "Online", "CreditCard")

#Exploring datasets
head(bank)
tail(bank)
summary(bank)
str(bank)

bank[,c(4,6,8,9,10,11,12)] <- lapply(bank[,c(4,6,8,9,10,11,12)],as.factor)

#Handling Missing values and negative values

colSums(bank[,-c(4,6,8,9,10,11,12)] < 0)
bank[bank$Experience < 0,"Experience"]  <- abs(bank[bank$Experience < 0,"Experience"])

any(is.na(bank))
colSums(is.na(bank))

bank_no_na <- na.omit(bank)

length(bank[bank$Personal.Loan == "1","Personal.Loan"])/nrow(bank)
length(bank[bank$Personal.Loan == "0","Personal.Loan"])/nrow(bank)


#Histograms
par(mfrow = c(2,3))
for(i in names(bank[-c(4,6,8,9,10,11,12)])){
  hist(bank[,i], xlab = names(bank[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(bank[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))


#Barplot
table1 <- bank_no_na %>% group_by(Family.Members,Personal.Loan) %>% summarise("Values" = n())
A <- ggplot(table1,aes(x = Personal.Loan, y = Values, fill = Family.Members), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)

table2 <- bank_no_na %>% group_by(Online,Personal.Loan) %>% summarise("Values" = n())
B <- ggplot(table2,aes(x = Personal.Loan, y = Values, fill = Online ), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)

table3 <- bank_no_na %>% group_by(Education,Personal.Loan) %>% summarise("Values" = n())
C <- ggplot(table3,aes(x = Personal.Loan, y = Values, fill = Education ), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)

table4 <- bank_no_na %>% group_by(Security.Account,Personal.Loan) %>% summarise("Values" = n())
D <- ggplot(table4,aes(x = Personal.Loan, y = Values, fill = Security.Account ), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)
table5 <- bank_no_na %>% group_by(CD.Account,Personal.Loan) %>% summarise("Values" = n())
E <- ggplot(table5,aes(x = Personal.Loan, y = Values, fill = CD.Account ), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)

table6 <- bank_no_na %>% group_by(CreditCard,Personal.Loan) %>% summarise("Values" = n())
G <- ggplot(table6,aes(x = Personal.Loan, y = Values, fill = CreditCard ), colour = "Black") + geom_bar(stat = "identity", position = "dodge")+                                                                              
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)
grid.arrange(A,B,C,D,E,G)

#Boxplot
H <- ggplot(bank) + geom_boxplot(aes(x=Personal.Loan, y = Age), color = c("Red","Green")) + ggtitle("Age vs Personal Loan")
I <- ggplot(bank) + geom_boxplot(aes(x=Personal.Loan, y = Income), color = c("Red","Green")) + ggtitle("Income vs Personal Loan")
J <- ggplot(bank) + geom_boxplot(aes(x=Personal.Loan, y = Mortgage), color = c("Red","Green")) + ggtitle("Mortgage vs Personal Loan")
K <- ggplot(bank) + geom_boxplot(aes(x=Personal.Loan, y = CCAvg), color = c("Red","Green"))+ ggtitle("CCavg vs Personal Loan")

grid.arrange(H,I,J,K)

#-------------------------------------------------------------------------------------

#Model Building

set.seed(101)
sample <- sample.split(bank$Personal.Loan, SplitRatio = 0.7)
train <- bank[sample == T, ]
test <- bank[sample == F,]

#Decision tree cart
set.seed(123)
tree <- rpart(train$Personal.Loan ~., data = train, control = rpart.control(minbucket = 8,xval = 10))       
library(rattle)
fancyRpartPlot(tree)
detach("package:rattle", unload = TRUE)
printcp(tree)
plotcp(tree)



#________________________________________________________________________________#

#Pruning of decision tree
best.cp <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
ptree <- prune(tree,cp = best.cp)
printcp(ptree)
library(rattle)
fancyRpartPlot(ptree)
detach("package:rattle", unload = TRUE)

#Cross Validation

#Predicting the training dataset
train$pred.prob.prune <- predict(ptree, newdata = train, type = "prob")[,"1"]
train$pred.class.prune <- ifelse(train$pred.prob.prune>0.7,1,0)


#Confusion Matrix
cm.train.prune <- confusionMatrix(train$Personal.Loan, as.factor(train$pred.class.prune))

Accuracy.train.prune <- cm.train.prune$overall[[1]]
Accuracy.train.prune 

class.err.train.prune <- 1 - Accuracy.train.prune
class.err.train.prune

Sensitivity.train.prune <- cm.train.prune$byClass[[1]]
Sensitivity.train.prune

Specificity.train.prune <- cm.train.prune$byClass[[2]]
Specificity.train.prune

#ROC/AUC
train.roc.pred.prune <- prediction(as.numeric(train$pred.class.prune), train$Personal.Loan)
perf.train.prune <- performance(train.roc.pred.prune,"tpr","fpr")
plot(perf.train.prune)

auc.train.prune<- performance(train.roc.pred.prune,"auc")
auc.train.prune <-auc.train.prune@y.values
auc.train.prune

#KS stat
ks.train.prune <- perf.train.prune@y.values[[1]] - perf.train.prune@x.values[[1]] 
ks.train.prune <- ks.train.prune[2]
ks.train.prune

#Gini
gini.train.prune <- ineq(train$pred.prob.prune,parameter = "gini")
gini.train.prune

#Concordance
library(InformationValue)
cord.train.prune <- Concordance(actuals = train$Personal.Loan,predictedScores = train$pred.prob.prune)
cord.train.prune <- cord.train.prune$Concordance
cord.train.prune
detach("package:InformationValue", unload = TRUE)

train.model.perf.prune <- t(data.frame(Accuracy.train.prune,class.err.train.prune,Sensitivity.train.prune,Specificity.train.prune,ks.train.prune,auc.train.prune,gini.train.prune,cord.train.prune))
rownames(train.model.perf.prune)<- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
train.model.perf.prune


#Predicting test set
test$pred.prob.prune <- predict(ptree, newdata = test, type = "prob")[,"1"]
test$pred.class.prune <- ifelse(test$pred.prob.prune>0.7,1,0)


#Model performance measure

#Confusion Matrx
cm.test.prune <- confusionMatrix(test$Personal.Loan, as.factor(test$pred.class.prune))

Accuracy.test.prune <- cm.test.prune$overall[[1]]
Accuracy.test.prune 

class.err.test.prune <- 1 - Accuracy.test.prune
class.err.test.prune

Sensitivity.test.prune <- cm.test.prune$byClass[[1]]
Sensitivity.test.prune

Specificity.test.prune <- cm.test.prune$byClass[[2]]
Specificity.test.prune


#ROC/AUC
test.roc.pred.prune<- prediction(as.numeric(test$pred.class),test$Personal.Loan)
perf.test.prune <- performance(test.roc.pred.prune, "tpr","fpr")
plot(perf.test.prune)

auc.test.prune <- performance(test.roc.pred.prune, "auc")
auc.test.prune <- auc.test.prune@y.values
auc.test.prune

#KS Stat
ks.test.prune <- perf.test.prune@y.values[[1]] - perf.test.prune@x.values[[1]]
ks.test.prune <- ks.test.prune[2]
ks.test.prune

#Gini

gini.test.prune <- ineq(test$pred.prob.prune, parameter = "gini")
gini.test.prune

#Concordance and Discordance Ratio
library(InformationValue)
cord.test.prune <- Concordance(actuals = test$Personal.Loan,predictedScores = test$pred.prob.prune)
cord.test.prune <- cord.test.prune$Concordance
cord.test.prune
detach("package:InformationValue", unload = TRUE)

test.model.perf.prune <- t(data.frame(Accuracy.test.prune,class.err.test.prune,Sensitivity.test.prune,Specificity.test.prune,ks.test.prune,auc.test.prune,gini.test.prune,cord.test.prune))
row.names(test.model.perf.prune) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
test.model.perf.prune

#Comparasion of model performance measure 

Model.perf.prune <- data.frame(train.model.perf.prune,test.model.perf.prune)
Model.perf.prune$Difference <- Model.perf.prune[,1] - Model.perf.prune[,2]
colnames(Model.perf.prune) <- c("Train","Test","Difference")
Model.perf.prune

#_________________________________________________________________________________________#

#Random Forest

bank_no_na[,c(4,6,8,9,10,11,12)] <- lapply(bank_no_na[,c(4,6,8,9,10,11,12)],as.factor)
colSums(is.na(bank_no_na))
library(caTools)
set.seed(101)
sampleRF <- sample.split(bank_no_na$Personal.Loan, SplitRatio = 0.7)
trainRF <- bank_no_na[sampleRF == T, ]
testRF <- bank_no_na[sampleRF == F,]

set.seed(123)
forest <- randomForest(Personal.Loan ~ ., data = trainRF, ntree = 501, mtry = 5, nodesize = 10, importance = TRUE, proximity = TRUE)
print(forest)
plot(forest)
import.var <- as.data.frame(importance(forest))
import.var

#Tuning the random forest
set.seed(123)
tune.forest <- tuneRF(x = trainRF[,-8], y = trainRF[,8], mtryStart = 5, stepFactor = 1.5,nodesize = 10,improve = 0.0001,ntreeTry = 301, importance = TRUE, plot = TRUE, trace = TRUE,doBest = TRUE)
print(tune.forest) 
import.var.tune <- as.data.frame(importance(tune.forest))
import.var.tune[order(import.var$MeanDecreaseGini, decreasing = T),]


#Predicting Training dataset
trainRF$pred.prob.tune <- predict(tune.forest, newdata = trainRF, type ="prob")[,"1"]
trainRF$pred.class.tune <- ifelse(trainRF$pred.prob.tune>0.7, 1, 0)

#Confusion Matrix
cm.rf.train.tune <- confusionMatrix(trainRF$Personal.Loan, as.factor(trainRF$pred.class.tune))

Accuracy.trainRF.tune <- cm.rf.train.tune$overall[[1]]
Accuracy.trainRF.tune 

class.err.trainRF.tune <- 1 - Accuracy.trainRF.tune
class.err.trainRF.tune

Sensitivity.trainRF.tune <- cm.rf.train.tune$byClass[[1]]
Sensitivity.trainRF.tune

Specificity.trainRF.tune <- cm.rf.train.tune$byClass[[2]]
Specificity.trainRF.tune

#ROC/AUC
train.roc.predRF <- prediction(as.numeric(trainRF$pred.class.tune),trainRF$Personal.Loan)
train.perf.RF <- performance(train.roc.predRF,"tpr","fpr")
plot(train.perf.RF)

train.auc.predRF <- performance(train.roc.predRF,"auc")
train.auc.predRF <- train.auc.predRF@y.values
train.auc.predRF <- train.auc.predRF[[1]]
train.auc.predRF

#KS_stat
train.ks.RF <- train.perf.RF@y.values[[1]] - train.perf.RF@x.values[[1]]
train.ks.RF <- train.ks.RF[2]
train.ks.RF


#Gini
train.gini.RF <- ineq(trainRF$pred.prob.tune, type ="Gini")
train.gini.RF

#Concordance
library(InformationValue)
train.cord.RF <- Concordance(predictedScores = trainRF$pred.prob.tune,actuals = trainRF$Personal.Loan)
train.cord.RF <- train.cord.RF[[1]]
train.cord.RF
detach("package:InformationValue", unload = TRUE)

trainRF.model.perf.tune <- t(data.frame(Accuracy.trainRF.tune,class.err.trainRF.tune,Sensitivity.trainRF.tune,Specificity.trainRF.tune,train.ks.RF,train.auc.predRF,train.gini.RF,train.cord.RF))
row.names(trainRF.model.perf.tune) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
trainRF.model.perf.tune

# Predicting testing dataset

testRF$pred.prob.tune <- predict(tune.forest, newdata = testRF, type ="prob")[,"1"]
testRF$pred.class.tune <- ifelse(testRF$pred.prob.tune>0.7, 1, 0)


#Confusion Matrix
cm.rf.test.tune <- confusionMatrix(testRF$Personal.Loan, as.factor(testRF$pred.class.tune))


Accuracy.testRF.tune <- cm.rf.test.tune$overall[[1]]
Accuracy.testRF.tune 

class.err.testRF.tune <- 1 - Accuracy.testRF.tune
class.err.testRF.tune

Sensitivity.testRF.tune <- cm.rf.test.tune$byClass[[1]]
Sensitivity.testRF.tune

Specificity.testRF.tune <- cm.rf.test.tune$byClass[[2]]
Specificity.testRF.tune

#ROC/AUC
test.roc.predRF <- prediction(testRF$pred.class.tune,testRF$Personal.Loan)
test.perf.RF <- performance(test.roc.predRF,"tpr","fpr")
plot(test.perf.RF)

test.auc.predRF <- performance(test.roc.predRF,"auc")
test.auc.predRF <- test.auc.predRF@y.values
test.auc.predRF <- test.auc.predRF[[1]]
test.auc.predRF

#KS_stat
test.ks.RF <- test.perf.RF@y.values[[1]] - test.perf.RF@x.values[[1]]
test.ks.RF <- test.ks.RF[2]
test.ks.RF

library(ineq)

test.gini.RF <- ineq(testRF$pred.prob.tune, type = "Gini")
test.gini.RF

#Concordance
library(InformationValue)
test.cord.RF <- Concordance(predictedScores = testRF$pred.prob.tune,actuals = testRF$Personal.Loan)
test.cord.RF <- test.cord.RF[[1]]
test.cord.RF
detach("package:InformationValue", unload = TRUE)


testRF.model.perf.tune <- t(data.frame(Accuracy.testRF.tune,class.err.testRF.tune,Sensitivity.testRF.tune,Specificity.testRF.tune,test.ks.RF,test.auc.predRF,test.gini.RF,test.cord.RF))
row.names(testRF.model.perf.tune) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
testRF.model.perf.tune

Model.perf.tuneRF <- data.frame(trainRF.model.perf.tune,testRF.model.perf.tune)
Model.perf.tuneRF$Difference <- Model.perf.tuneRF[,1] - Model.perf.tuneRF[,2]
colnames(Model.perf.tuneRF) <- c("TrainRF","TestRF","DifferenceRF")
Model.perf.tuneRF


##Model Performance measure of the Random Forest and Cart 

Combine.Model.Perf <- data.frame(Model.perf.prune,Model.perf.tuneRF)
colnames(Combine.Model.Perf) <- c("Cart.Train","Cart.Test","Difference.Cart","RandomForest.Train","RandomForest.Test","Difference.RF")
Combine.Model.Perf



