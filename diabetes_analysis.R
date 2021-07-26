# 0 = False, 1 = True 

# Reading
#https://www.kaggle.com/uciml/pima-indians-diabetes-database/activity
setwd("Documents/Kaggle/Pima Indians Diabetes Prediction /")
db = read.csv("diabetes.csv")
library(ggplot2)
library(gridExtra)
library(class)
library(tree)
library(caret)
library(e1071)
library(randomForest)
library(neuralnet)
library(xgboost)
library(pROC)


attach(db)

head(db)

# ____________________ INTRODUCTION ________________________
# Lets perform some explanatory data analysis to see the behaviour of the data which will help in preprocessing 
# View the datamframes column and rows 
dim(db)

# View the attributes type within the dataframe
str(db)

# Numerical description of the dataframe
summary(db)

################## Summary: (DO LATER)

# Data Cleaning 
# Rename the attribute "DiabetesPedigreeFunction for convenience
colnames(db)[7] = "DPF"
head(db)

# No missing values found 
colSums(is.na(db))
colSums(db=="")

db$Outcome = as.factor(db$Outcome)

# ____________________ OUTLIERS________________________

# Outlier detection and normalising
outlier_norm <- function(x){
  qntile <- quantile(x, probs=c(.25, .75))
  caps <- quantile(x, probs=c(.05, .95))
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qntile[1] - H)] <- caps[1]
  x[x > (qntile[2] + H)] <- caps[2]
  return(x)
}

# Creation of bar graph and box plot to detect outliers within attributes 
outlier_detect <- function(var, c, b){
  p1 = ggplot(data=db, aes(x=var)) +
    geom_histogram(aes(x=var, y=(..count..)/sum(..count..)), binwidth=b, fill=c, col="white") + 
    geom_density() +
    xlab(paste(deparse(substitute(var)), "Class")) +
    ylab("Frequency") +
    ggtitle(paste("Distribution of ", deparse(substitute(var))))

  p2 = ggplot(data=db, aes(y=var)) +
    stat_boxplot(geom="errorbar", width=0.2) +
    geom_boxplot(fill=c) + ylab(deparse(substitute(var))) +
    ggtitle(paste("Five Point Summary (", deparse(substitute(var)), ")"))
  
  grid.arrange(p1, p2, ncol=2)
}

# Previously, it has shown that some instances within Preganicies attribute has a
# maximum of 17, which does not make too much sense
# To show if there are any outliers, a boxplot is created to easily visualise

# ____________________ Pregnancies ________________________
# Pregnancies outliers detection
outlier_detect(Pregnancies, "red", 1)

db$Pregnancies = outlier_norm(db$Pregnancies)
attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(Pregnancies, "red", 1)

# ____________________ Glucose ________________________
# Glucose outliers detection
outlier_detect(Glucose, "orange", 7)

# This data has a normal distribution, and thus treating the outliers with a mean value is appropriate
db$Glucose[db$Glucose==0] = mean(db$Glucose)
attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(Glucose, "orange", 7)

# ____________________ BloodPressure ________________________
# BloodPressure outliers detection
outlier_detect(BloodPressure, "gold", 5)

# This data has a normal distribution, however there are a lot of instances with the value 0
# Therefore, using a median to treat these instances would be more appropriate than the mean as it would get dragged out 
db$BloodPressure[db$BloodPressure==0] = median(db$BloodPressure)
# We then can normalise the outliers using our function
db$BloodPressure = outlier_norm(db$BloodPressure)

attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(BloodPressure, "gold", 3)

# ____________________ SkinThickness ________________________
# SkinThickness outliers detection
outlier_detect(SkinThickness, "forestgreen", 6)

# Closer to normal distribution
db$SkinThickness[db$SkinThickness==0] = mean(db$SkinThickness)
# Normalise the outliers using our function
db$SkinThickness = outlier_norm(db$SkinThickness)

attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(SkinThickness, "forestgreen", 3)

# ____________________ BMI ________________________
# BMI outliers detection
outlier_detect(BMI, "salmon", 6)

# Closer to normal distribution
db$BMI[db$BMI==0] = mean(db$BMI)

attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(BMI, "salmon", 6)

# ____________________ DPF ________________________
# BMI outliers detection
outlier_detect(DPF, "mediumpurple", 0.3)

# There appears to be a lot of outliers on the higher end
db$DPF = outlier_norm(db$DPF)
attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(DPF, "mediumpurple", 0.3)

# ____________________ Age ________________________
# Age outliers detection
outlier_detect(Age, "tan3", 6)

# Normalise the outliers using our function
db$Age = outlier_norm(db$Age)

attach(db) # Reattach the dataset

# Verifying the treatment of the outliers
outlier_detect(Age, "tan3", 6)

# ____________________ Redundant Features ________________________
# Remove attributes with an absolute correlation of 0.75 or higher
feat.cor = findCorrelation(cor(db[,1:8]), cutoff=0.5)
colnames(db[feat.cor[1]])
colnames(db[feat.cor[2]])

# This relates to attribute 6 and 8, which is BMI and Age
# Therefore, we would ideally want to remove one or the other
# This is done to make the learning algorithm faster, decrease harmful bias 
# And interpretability of the model 


# ____________________Feature Importance ________________________
# Through using the library caret, we are able to construct a Learning Vector Quanitzation (LVQ) model.
# Where the varImp is then used to estimate the variable importance
control = trainControl(method="repeatedcv", number=10, repeats=3)
model = train(Outcome~., data=db, method="lvq", preProcess="scale", trControl=control)
importance = varImp(model, scale=FALSE)
plot(importance)

# ____________________Feature Selection ________________________
# Feature selection is used to identify those attributes that are and are not required to build an accurate model 
# We will be using an automatic method for feature selection called Recursive Feature Elimination or RFE
control = rfeControl(functions=rfFuncs, method="cv", number=10)
feat.res = rfe(db[,1:8], db[,9], sizes=c(1:8), rfeControl=control)

# chosen features
predictors(feat.res)

# plot the results
plot(feat.res, type=c("g","o"))
feat.res$results

# ____________________ ANALYSIS ________________________
# Subset the data into train and test 
set.seed(1)
train.row = sample(1:nrow(db), 0.7*nrow(db))
db.train = db[train.row,]
db.test = db[-train.row,]
train.target = db.train$Outcome
test.target = db.test$Outcome

# ____________________ KNN ________________________
acc <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

# We have 537 observations within our train set
# with the square root of 537 being 23.17, we will create two models
# One with the "k" value of 23 and the other being 24
length(train.target)

knn.res.23 = table(knn(db.train, db.test, cl=train.target, k=23), test.target)

knn.res.24 = table(knn(db.train, db.test, cl=train.target, k=24), test.target)

paste("Accuracy on test set (23):", acc(knn.res.23), "%")
paste("Accuracy on test set (24):", acc(knn.res.24), "%")

# ____________________ KNN (Elbow Method) & Accuracy Graph ________________________
# We do this to check whether or not the values chosen has the highest accuracy
db.knn.opt = 1 
for (i in 1:25){
  db.knn = knn(db.train, db.test, cl=train.target, k=i)
  db.knn.opt[i] = acc(table(db.knn, test.target))
}

plot(db.knn.opt, type="l", xlab="K- Value",ylab="Accuracy %", 
     main="Accuracy of KNN model for 'K' values from 1 - 25")

paste("The K- Value that has the highest accuracy is:", which.max(db.knn.opt))
paste("This produced an accuracy of:", db.knn.opt[which.max(db.knn.opt)])
db.knn.final = knn(db.train, db.test, cl=train.target, k=which.max(db.knn.opt), "%")
# By using the Elbow Method and Accuracy Graph, it can be seen that the intial K- Value chosen 
# did not produce the highest accuracy. Therefore, it is important to further prove the initial result

# ____________________ Decision Tree ________________________
db.tree = tree(Outcome~., data = db.train)
# Basic idea of model performance (tree)
summary(db.tree) 

plot(db.tree)
text(db.tree, pretty = 0) # Fix text
title("Predicting diabetes outcome")

tpredict = predict(db.tree, db.test, type = "class")
tree.res = table(observed = db.test$Outcome, predicted = tpredict)
paste("Accuracy of Decision Tree:", acc(tree.res), "%")

# ____________________ Random Forest ________________________
db.rf = randomForest(Outcome ~., data = db.train, ntree=100)

rfpredict = predict(db.rf, db.test)
rf.res = table(observed = db.test$Outcome, predicted = rfpredict)
paste("Accuracy of Random Forest:", acc(rf.res), "%")

plot(randomForest(Outcome ~., data = db.train, keep.forest = FALSE, ntree=100))

# ____________________ Support Vector Machines (SVM) ________________________
db.svm = svm(Outcome ~ ., data = db.train, kernel = "linear", cost = 10, scale = FALSE)

svmpredict = predict(db.svm, db.test)
svm.res = table(observed = db.test$Outcome, predicted = svmpredict)
paste("Accuracy of SVM:", acc(svm.res), "%")

# ____________________ Artificial Neural Network (ANN) ________________________
db.ann = neuralnet(Outcome ~., db.train, hidden = 2, act.fct="tanh")

plot(db.ann, rep = "best")

# predicting of neural network 
ann.pred = compute(db.ann, db.test)

ann.predr = round(ann.pred$net.result,0)

ann.res = table(observed = db.test$Outcome, predicted = ann.predr[,2])
paste("Accuracy of ANN:", acc(ann.res), "%")

# ____________________ XGBoost Classifier ________________________
# Label conversion
tr.outcome = db.train$Outcome
ts.outcome = db.test$Outcome

tr.label = as.integer(tr.outcome)-1
ts.label = as.integer(ts.outcome)-1

db.train$Outcome = NULL
db.test$Outcome = NULL

# Create xgb.DMatrix objects
xgb.train = xgb.DMatrix(data=as.matrix(db.train), label=tr.label)
xgb.test = xgb.DMatrix(data=as.matrix(db.test), label=ts.label)

# Define the main parameters 
num_class = length(levels(tr.outcome))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Train the model 
# Verbose can be set to 1 to see the results 
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

xgb.fit

xgb.predict = predict(xgb.fit, as.matrix(db.test), reshape=T)
xgb.predict = as.data.frame(xgb.predict)
xgb.predict = as.numeric(as.logical(xgb.predict[,2] > 0.5))
xgb.res = table(observed = xgb.predict, predicted = ts.outcome)

paste("Accuracy of XGBoost:", acc(xgb.res), "%")

# ____________________ ROC & AUC ________________________   
# ROC = Receiver Operating Characteristics
# AUC = Area Under Curve 

# KNN ROC
db.knn.final = knn(db.train, db.test, cl=train.target, k=which.max(db.knn.opt))
kpred = prediction(as.numeric(db.knn.final), ts.outcome)
knnroc = performance(kpred, "tpr", "fpr")
knnauc = performance(kpred, "auc") 

# Decision Tree ROC
tpred = predict(db.tree, db.test, type = "vector")
treepred = prediction(tpred[,2], ts.outcome)
treeroc = performance(treepred, "tpr", "fpr")
treeauc = performance(treepred, "auc")

# Random Forest ROC
rfpred = predict(db.rf, db.test, type = "prob")
randpred = prediction(rfpred[,2], ts.outcome)
randroc = performance(randpred, "tpr", "fpr")
randauc = performance(randpred, "auc")

# SVM ROC 
spred = predict(db.svm, db.test, type="response")
svmpred = prediction(as.numeric(spred)-1, ts.outcome)
svmroc = performance(svmpred, "tpr", "fpr")
svmauc = performance(svmpred, "auc")

# XGBoost ROC
xpred = prediction(xgb.predict, ts.outcome)
xgbroc = performance(xpred, "tpr", "fpr")
xgbauc = performance(xpred, "auc")

# ROC Plot 
plot(knnroc, col = "red")
plot(treeroc, add = TRUE, col = "blue")
plot(randroc, add = TRUE, col = "green3")
plot(svmroc, add = TRUE, col = "orange")
plot(xgbroc, add = TRUE, col = "mediumpurple")


abline(0,1, lty = 2)
title("ROC Curve for Each Classifier")
legend(0, 1.02, c("KNN", "Decision Tree", "Random Forest", "SVM", "XGBoost", "Reference Line"),
       lty = c(1,1,1,1,1,2), col = c("red", "blue", "green3", "orange", "mediumpurple", "black"), cex = 0.75)

# AUC Table
model.auc = cbind(lapply(c(knnauc@y.values, treeauc@y.values, randauc@y.values, svmauc@y.values, xgbauc@y.values), as.numeric)) 
rownames(model.auc) = c("KNN", "Decision Tree", "Random Forest", "SVM", "XGBoost")
colnames(model.auc) = "AUC (%)"
model.auc = as.data.frame(model.auc)
model.auc

# Accuracy Table
model.acc = rbind(db.knn.opt[which.max(db.knn.opt)], acc(tree.res), acc(rf.res), acc(svm.res), acc(ann.res),acc(xgb.res))
rownames(model.acc) = c("KNN", "Decision Tree", "Random Forest", "SVM", "ANN", "XGBoost")
colnames(model.acc) = "AUC (%)"
model.acc = as.data.frame(model.acc)
model.acc

# Overall, the best performing models are Random Forest, Decision Trees and XGBoost