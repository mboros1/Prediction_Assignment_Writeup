---
title: "Prediction Assignment Writeup"
author: "Martin Boros"
date: "April 29, 2016"
output: html_document
---
##Load packages, load the datasets, set the seed.
  
```{r, message = FALSE} 
library(caret)
library(randomForest)
library(plyr)
set.seed(123)
workoutdata = read.csv("pml-training.csv")
testset = read.csv("pml-testing.csv")
```

##Cross Validation
The training data set is broken up into a set for training the machine learning algorithm and testing its efficacy. For cross-validation, the "out-of-bag"
estimates method was used with a 10-fold cross-validation repeated 10 times.

```{r}
inTrain = createDataPartition(workoutdata$classe, p = 3/4)[[1]]
training = workoutdata[inTrain,]
testing = workoutdata[-inTrain,]
modelcontrol = trainControl(method = "oob", number = 10, repeats = 10)
```



##Setting up the Predictive Model
While there were 159 available features, a huge majority of them contained missing data and data of questionable value. Ignoring these features cut the
available number of features down to about 59. Using these 59 features, the data set became overfitted to the training data set and performed poorly on
the test data set. I was able to create a highly effective model using 6 features: the user name, part of the timestamp, and 3 pieces of the belt device's
data. While this gave fantastic results and seemed to be in line with the question asked, it felt like it was "cheating" since the timestamp tracked when
each exercise was performed, making it very predictive while not actually giving any useful insights on how the exercise was performed.
Because of this, I created a second model whose features were the user name, and data from the belt device and forearm device. While it was not quite as
predictive as the model using the timestamp, it seemed more inline with the spirit of the test that was performed.

```{r}
model = randomForest(classe ~ user_name + raw_timestamp_part_1 + roll_belt + pitch_belt + yaw_belt + total_accel_belt, data = training, trainControl = modelcontrol)
modelfair = randomForest(classe ~ user_name + roll_belt + pitch_belt + yaw_belt + total_accel_belt + gyros_forearm_y + gyros_forearm_z + gyros_forearm_x + accel_forearm_x + accel_forearm_y + accel_forearm_z + magnet_forearm_x + magnet_forearm_y + magnet_forearm_z, data = training, trainControl = modelcontrol)
```

For the predictive models, I used the Random Forest method. The computation time was quite low, and had the highest accuracy from the methods outlined in
the Machine Learning course.


```{r}
p = predict(model, testing[,-160])
confusionMatrix(p, testing$classe)
error = 1 - .9973
error

```

Here we can see the accuracy of the model on the test set is 99.73%, giving us an estimated out of sample error of 0.23%.



```{r}
pfair = predict(modelfair, testing[,-160])
confusionMatrix(pfair, testing$classe)
error = 1 - .9753
error

```

This model is less accurate, but more "fair". The estimated out of sample error is 2.47%.


##Final comparison

```{r}
pfinal = predict(model, testset)
pfairfinal = predict(modelfair, testset)
confusionMatrix(pfinal, pfairfinal)
```

Both models are run on the original test set, and given the relatively small sample size (N=20), both tests were more then adequate to predict the outcomes
perfectly.