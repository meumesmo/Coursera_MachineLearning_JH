---
title: "Final Project - Machine Learning - Coursera"
author: "Pedro Lealdino Filho"
date: "3/11/2018"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



## Project Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.


```r
library(e1071)
library(caret)
library(rpart.plot)
library(rpart)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(corrplot)
library(gbm)
library(dplyr)
library(lubridate)
library(lucr)
library(data.table)
library(ggplot2)
library(tidyverse)
library(knitr)
```

### The Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

We start getting the data :

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
test <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```

### The Goal

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Exploring the Data

Let's get a look on how the data looks like:

```r
str(train)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```
Woosh! There are a lot of NA variables here.

```r
tempTrain <- train
tempTrain <- purrr::map_dbl(tempTrain, function(x) {round((sum(is.na(x))/ length(x)) * 100, 1)})
tempTrain <- tempTrain[tempTrain > 0]

data.frame(miss = tempTrain, var = names(tempTrain), row.names = NULL ) %>%
  ggplot(aes(x=reorder(var, -miss), y=miss)) + 
  geom_bar(stat='identity', fill='red') +
  labs(x='', y='% missing', title='Percent missing data by feature') +
  theme(axis.text.x=element_text(angle=90, hjust=1))
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-4-1.png)<!-- -->


First of all let's create our partitions:

```r
inTrain <- createDataPartition(train$classe, p = 0.8, list = FALSE)
myTrain <- train[inTrain,]
myTest <- train[-inTrain,]
```

## Cleaning the Data
Removing the Near Zero Variance

```r
nearZero <- nearZeroVar(myTrain, saveMetrics = TRUE)
myTrain <- myTrain[,nearZero$nzv == FALSE]

nearZero <- nearZeroVar(myTest, saveMetrics = TRUE)
myTest <- myTest[, nearZero$nzv == FALSE]
```

Let's remove the first column of the myTrain dataset because it's not interesting to the problem

```r
myTrain <- myTrain[c(-1)]
```

Cleaning varibles with more than 60% NA's:

```r
training <- myTrain
for(i in 1:length(myTrain)){
  if(sum(is.na(myTrain[,i]))/nrow(myTrain) >= .6) {
    for(j in 1:length(training)){
      if(length(grep(names(myTrain[i]), names(training)[j])) == 1){
        training <- training[, -j]
      }
    }
  }
}

myTrain <- training
```

Transform the myTest and test data sets:

```r
clean1 <- colnames(myTrain)
clean2 <- colnames(myTrain[, -58])
myTest <- myTest[clean1]
test <- test[clean2]
```


```r
for (i in 1:length(test)) {
  for (j in 1:length(myTrain)){
    if(length(grep(names(myTrain[i]), names(test)[j])) == 1) {
      class(test[j]) <- class(myTrain[i])
    }
  }
}

test <- rbind(myTrain[2, -58], test)
test <- test[-1, ]
```

## Decision Trees

```r
modFit1 <- rpart(classe ~ ., data = myTrain, method = "class")
fancyRpartPlot(modFit1)
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-11-1.png)<!-- -->


```r
predictions1 <- predict(modFit1, myTest, type = "class")
cmtree <- caret::confusionMatrix(predictions1, myTest$classe)
cmtree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1070   36    1    2    0
##          B   33  619   34   36    0
##          C   13   97  631   67   27
##          D    0    7   11  426   46
##          E    0    0    7  112  648
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8652          
##                  95% CI : (0.8541, 0.8757)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8294          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9588   0.8155   0.9225   0.6625   0.8988
## Specificity            0.9861   0.9674   0.9370   0.9805   0.9628
## Pos Pred Value         0.9648   0.8573   0.7557   0.8694   0.8449
## Neg Pred Value         0.9837   0.9563   0.9828   0.9368   0.9769
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2728   0.1578   0.1608   0.1086   0.1652
## Detection Prevalence   0.2827   0.1840   0.2128   0.1249   0.1955
## Balanced Accuracy      0.9724   0.8915   0.9298   0.8215   0.9308
```


```r
plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix : Accuracy = ", round(cmtree$overall['Accuracy'], 4)))
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

## Random Forest

```r
modFit2 <- randomForest(classe ~., data = myTrain)
predictions2 <- predict(modFit2, myTest, type = "class")
cmrf <- confusionMatrix(predictions2, myTest$classe)
cmrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  758    0    0    0
##          C    0    0  684    2    0
##          D    0    0    0  640    0
##          E    0    0    0    1  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   1.0000   0.9953   1.0000
## Specificity            0.9996   1.0000   0.9994   1.0000   0.9997
## Pos Pred Value         0.9991   1.0000   0.9971   1.0000   0.9986
## Neg Pred Value         1.0000   0.9997   1.0000   0.9991   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1744   0.1631   0.1838
## Detection Prevalence   0.2847   0.1932   0.1749   0.1631   0.1840
## Balanced Accuracy      0.9998   0.9993   0.9997   0.9977   0.9998
```

```r
plot(modFit2)
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

```r
plot(cmrf$table, col = cmrf$byClass, main = paste("Random Forest Confusion Matrix : Accuracy = ", round(cmrf$overall['Accuracy'], 4)))
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

## Generalized Boosted Regression

```r
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

modFit3 <- train(classe ~., data = myTrain, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)

modFit3$finalModel

modFit3f <- modFit3$finalModel

predictions3 <- predict(modFit3, newdata = myTest)
cmgbm <- confusionMatrix(predictions3, myTest$classe)
cmgbm
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 79 predictors of which 43 had non-zero influence.
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    4    0    0    0
##          B    0  754    0    0    0
##          C    0    1  681    4    0
##          D    0    0    3  637    1
##          E    0    0    0    2  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9937, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9934   0.9956   0.9907   0.9986
## Specificity            0.9986   1.0000   0.9985   0.9988   0.9994
## Pos Pred Value         0.9964   1.0000   0.9927   0.9938   0.9972
## Neg Pred Value         1.0000   0.9984   0.9991   0.9982   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1922   0.1736   0.1624   0.1835
## Detection Prevalence   0.2855   0.1922   0.1749   0.1634   0.1840
## Balanced Accuracy      0.9993   0.9967   0.9970   0.9947   0.9990
```


```r
plot(modFit3, ylim = c(0.9, 1))
```

![](Final_Project_ML_JH_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

## Predict Results on the Test Data

```r
prediction <- predict(modFit2, test, type = "class")
submit <- data.frame(portfolio_id = row.names(test), return =  prediction)
write.csv(submit, file = "Prediction.csv", row.names = FALSE)
```
