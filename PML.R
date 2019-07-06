#Download the data and then read the csv files

training <- read.csv('F:/pml-training.csv')

testing <- read.csv('F:/pml-testing.csv')

#load the CARET (CLASSIFICATION AND REGRESSION TRAINING) library

library(caret)     

#check the enitre dataset together

str(training)

#check the column names from the training dataset
colnames(training)

head(training$classe,5)

#The "classe" variable is our target variable which has 5 levels namely , A,B,C,D AND E
#On observing the variable we can deduce that this is a supervised macine problem and we
#can use regression and classification models to predict the outcome


#we observe lots of missing values in the dataset and so we must pre-process the data
#prior to using it in out machine kearning algorithm

#lets check the total missing values
sum(is.na(training))

#and the missing values pretaining to each variable
na_count <-sapply(training, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)

print(na_count)

#we see that the columns with missing values have more than 90% of missing values
#so we remove them completely

index_remCol <- which(colSums(is.na(training) | training =="")>0.9*dim(training)[1]) 
training_clean <- training[,-index_remCol]

#we remove first seven columns as well since they are not useful in our prediction
training_clean <- training_clean[,-c(1:7)]

dim(training_clean)

dim(testing)

#apply it to test set

index_remCol <- which(colSums(is.na(testing) | testing =="")>0.9*dim(testing)[1]) 
testing_clean <- testing[,-index_remCol]

testing_clean <- testing_clean[,-c(1:7)]

#now we partition training data to train and test, while we keep the above testing for validation
set.seed(1234) 

inTrain <- createDataPartition(training_clean$classe, p = 0.7, list = FALSE)
trainData <- training_clean[inTrain, ]
testData <- trainData[-inTrain, ]

dim(trainData)
dim(testData)


cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))




#model building

library(rpart)

#we can do basic parameter tuning to improve our models accuracy
#such as K-fold cv , leave one out , radom subsampling etc

fitControl <- trainControl(method = "cv", number = 5)

modelclasstree <- train(classe ~ . ,data = trainData, method = "rpart", trControl = fitControl)

#prediction
pred <- predict(modelclasstree, newdata = testData)

confusionMatrix(pred , testData$classe)

plot(modelclasstree)

#using random forest model

library(randomForest)

modelRF <- randomForest(classe ~ . , data = trainData)

#modelRF1 <- train(classe ~ ., data = trainData, method = "rf", ntree = 50,
#                 verbose=FALSE)


#prediction
predrf <- predict(modelRF, newdata = testData)

confusionMatrix(predrf , testData$classe)

names(modelRF$finalModel)

#plot of random forest 
plot(modelRF)

plot(predrf)


#trying the gradient boosted method

modeldiagGBM <- gbm(classe ~. ,data = trainData,distribution = "gaussian", n.trees = 100)

gbm()
#prediction


library(gbm)

predGBM <- predict.gbm(modeldiagGBM, newdata =testData ,n.trees = modeldiagGBM$n.trees )

u <- union(predGBM, testData$classe)
t <- table(factor(predGBM, u), factor(testData$classe, u))
confusionMatrix(t)

confusionMatrix(predGBM,testData$classe)






#apply best model to the initial testing data
#so best model is Randm Forest
Results <- predict(modelRF, newdata= testing_clean)
Results