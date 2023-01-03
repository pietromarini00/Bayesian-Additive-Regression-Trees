setwd("~/usr/dresktop/dir1")

library(tidyverse)
library(caTools)
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization

library(olsrr)        # OLS
library(BayesTree)
library(randomForest)
library(rpart)


# let's read the CSV and see the variables and their type
LOS = data.frame(read.csv("data_short.csv"))
str(LOS)
waview(LOS)

# drop the columns we don't know the meaning of
df = subset(LOS, select = -c(discharged,vdate,facid,rcount))

# create dummy variable for gender
df$gender = factor(df$gender, 
                   levels = c('F','M'), 
                   labels = c(0,1))
df$gender  = as.factor(df$gender)

# look at the distribution of length of stay
hist(df$lengthofstay, main="Length of Stay",xlab="Length of Stay")

# Negative Exponential, we can turn it into Normal by taking the log
df$log_len = log(df$lengthofstay)
hist(df$log_len, main="Length of Stay",xlab="Length of Stay",breaks=8)

set.seed(42)
split = sample.split(df, SplitRatio = 0.8)# returns true if observation goes to the Training set and false if observation goes to the test set.

#Creating the training set and test set separately
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Train test split
y_train  = training_set$log_len
X_train = subset(training_set, select = -c(log_len,lengthofstay))
y_test  = test_set$log_len
X_test = subset(test_set, select = -c(log_len,lengthofstay))


# ======================================================================================
# Ordinary Least Squares

model <- lm(y_train ~ ., data = X_train)

y_hat_lm <- predict(model,X_test)

lm_MSE = sum((y_hat_lm-y_test)^2)/length(y_hat_lm)

lm_MSE ## 0.3805606

# ======================================================================================
# Decision Tree

DecisionTree <- rpart(y_train ~ ., method = "anova", data = X_train)

y_hat = predict(DecisionTree, X_train, method = "anova")
mse_DTree = sum((y_train-y_hat)^2)/length(y_train)

mse_DTree    #Train  ->  0.361

y_hat = predict(DecisionTree, X_test, method = "anova")
mse_DTree = sum((y_test-y_hat)^2)/length(y_test)

mse_DTree   #Test  ->  0.364

plot(DecisionTree)
plotcp(DecisionTree)
residuals.rpart(DecisionTree)
residuals(DecisionTree)
plotcp(DecisionTree)

# ======================================================================================
# Gradient Boosted Model

gbm.fit <- gbm(
  formula = y_train ~ .,
  distribution = "gaussian",
  data = X_train,
  n.trees = 1000,
  interaction.depth = 1,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
  )  

# get MSE and compute RMSE
MSE_gbm <- min(gbm.fit$cv.error)

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")

# tuning
# train GBM model
gbm.fit2 <- gbm(
  formula = y_train ~ .,
  distribution = "gaussian",
  data = X_train,
  n.trees = 1000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit2$cv.error)

# get MSE and compute RMSE
MSE_gbm2 <- gbm.fit2$cv.error[min_MSE]   ## [1] 0.2905935

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit2, method = "cv")

# ======================================================================================
# XGBoost

#train = subset(training_set, select = -c(lengthofstay))
#test = subset(test_set, select = -c(lengthofstay))


train.matrix = as.matrix(subset(training_set, select = -c(lengthofstay)))
mode(train.matrix) = "numeric"
test.matrix = as.matrix(subset(test_set, select = -c(lengthofstay)))
mode(test.matrix) = "numeric"

xgb_train = xgb.DMatrix(data = train.matrix[,1:23], label = train.matrix[,24])
xgb_test = xgb.DMatrix(data = test.matrix[,1:23], label = test.matrix[,24])

#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 220)

xgb.ggplot.deepness(model)

#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 220, verbose = 0)
summary(model_xgboost)

#use model to make predictions on test data
pred_y = predict(model_xgboost, xgb_test)
sum((y_test - pred_y)^2)/length(y_test) #mse - Mean Squared Error

y_test_mean = mean(y_test)

param <- list("objective" = "reg:squarederror",    # binary classification
              "eval_metric" = "rmse",    # evaluation metric
              "nthread" = 8,   # number of threads to be used
              "max_depth" = 16,    # maximum depth of tree
              "eta" = 0.2,    # step size shrinkage
              "gamma" = 0,    # minimum loss reduction
              "subsample" = 1,    # part of data instances to grow tree
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree
              "min_child_weight" = 12)  # minimum sum of instance weight needed in a child

xgboost.cv <- xgb.cv(param=param, data=train.matrix[,1:23], label = train.matrix[,24],
                     nfold=5, nrounds=100, prediction=TRUE, verbose=T)

xgboost.cv$evaluation_log

mse_test_xgb = xgboost.cv$evaluation_log[,'test_rmse_mean']^2
mse_train_xgb = xgboost.cv$evaluation_log[,'train_rmse_mean']^2

ggplot(model$evaluation_log, aes(y = MSE, x = iter)) +
  geom_line(aes(y = test_rmse^2, colour = "green")) +
  geom_line(aes(y = train_rmse^2, colour = "black"))
#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 220, verbose = 0)
summary(model_xgboost)
#use model to make predictions on test data
pred_y = predict(model_xgboost, xgb_test)

mean((y_test - pred_y)^2)              #mse - Mean Squared Error
x = 1:length(y_test)                   # visualize the model, actual and predicted data
plot(x, y_test, col = "red", type = "l")
lines(x, pred_y, col = "blue", type = "l")
legend(x = 1, y = 38,  legend = c("original test_y", "predicted test_y"),
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))

# ======================================================================================
# Random Forest

rf <- randomForest(y_train ~ ., data=X_train)

plot(rf)

rf_mse_train = rf$mse[500]

predict(rf,X_test)
y_hat = predict(rf,X_test)

rf_test_mse = sum((y_hat-y_test)^2)/length(y_test)

rf_test_mse    ## 0.2939061

y_rf_train = predict(rf,X_train)
rf_train_mse = sum((y_rf_train-y_train)^2)/length(y_train)   # 0.2918909

# running time

rf$mse

ggplot(data.frame(rf$mse), aes(y = MSE, x = iter)) +
  geom_line(aes(y = data.frame(rf_test_rmse^2), colour = "test")) +
  geom_line(aes(y = data.frame(train_rmse^2), colour = "train"))

# ======================================================================================
# Bayesian Additive Regression - bartMachine

y_train  = training_set$log_len
X_train = subset(training_set, select = -c(log_len,lengthofstay))
y_test  = test_set$log_len
X_test = subset(test_set, select = -c(log_len,lengthofstay))

options(java.parameters = "-Xmx30g")

library(bartMachine)

num_cores = 4
set_bart_machine_num_cores(num_cores)
bart_machine_num_cores()

# default bartMachine
# running time single-core: {start: 10:02:00, end:10:34:00}
# running time four-cores: {start: 15:27:00, end:}

bart_machine = bartMachine(X_train,y_train, num_trees = 200,
                           num_burn_in = 500,
                           num_iterations_after_burn_in = 1000,
                           alpha = 0.95,
                           beta = 2,
                           k = 2,
                           q = 0.9,
                           nu = 3.0,)


plot_convergence_diagnostics(bart_machine)


y_hat_bartMachine = bart_predict_for_test_data(bart_machine, X_test, y_test, prob_rule_class = NULL)



# MSE 0.28512438807

# aggressive bM

time0 = Sys.time()

bart_machine = bartMachine(X_train,y_train, num_trees = 200,
                           num_burn_in = 500,
                           num_iterations_after_burn_in = 1000,
                           alpha = 0.95,
                           beta = 2,
                           k = 2,
                           q = 0.99,
                           nu = 3.0)
time1 = Sys.time()

y_hat_aggressive = bart_predict_for_test_data(bart_machine, X_test, y_test, prob_rule_class = NULL)

time2 = Sys.time()

timedelta = time2-time0

# 0.2846242

# conservative bM

bart_machine = bartMachine(X_train,y_train, num_trees = 200,
                           num_burn_in = 500,
                           num_iterations_after_burn_in = 1000,
                           alpha = 0.95,
                           beta = 2,
                           k = 2,
                           q = 0.75,
                           nu = 10.0)

y_hat_conservative = bart_predict_for_test_data(bart_machine, X_test, y_test, prob_rule_class = NULL)


investigate_var_importance(bart_machine, num_replicates_for_avg = 5, plot=TRUE, num_var_plot = 23)

interaction_investigator(bart_machine, plot=TRUE, num_replicates_for_avg = 5,
                         num_trees_bottleneck = 20, num_var_plot = 50, cut_bottom = NULL,
                         bottom_margin = 15)

check_bart_error_assumptions(bart_machine,hetero_plot='yhats')


#cov_importance_test

get_sigsqs(bart_machine,after_burn_in=FALSE,plot_CI=TRUE)

plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)

plot_y_vs_yhat(bart_machine, Xtest=X_test, ytest=y_test, prediction_intervals = TRUE)



plot_convergence_diagnostics(bart_machine)

rmse_by_num_trees(bart_machine,plot = TRUE)

VS = var_selection_by_permute_cv(bart_machine, num_permute_samples=5)


# see if informed priors perform better
bart_informed = bartMachine(X_train,y_train, num_trees = 200,
                                           num_burn_in = 500,
                                           num_iterations_after_burn_in = 1000,
                                           alpha = 0.95,
                                           beta = 2,
                                           k = 2,
                                           q = 0.99,
                                           nu = 3.0,
                                           cov_prior_vec = prior)

# ======================================================================================
# Bayesian Additive Regression - BayesTree

y_train  = training_set$log_len
X_train = subset(training_set, select = -c(log_len,lengthofstay))
y_test  = test_set$log_len
X_test = subset(test_set, select = -c(log_len,lengthofstay))

BART <- bart( X_train, y_train, X_test, 
              sigest=NA, sigdf=3, sigquant=.90,
              k=2.0,
              power=2.0, base=.95,
              binaryOffset=0,
              ntree=200,
              ndpost=1000, nskip=100,
              printevery=100, keepevery=1, keeptrainfits=TRUE,
              usequants=FALSE, numcut=100, printcutoffs=0,
              verbose=TRUE)

#citation("BayesTree")

y_hat_BART = BART$yhat.test.mean

MSE_BART = sum((y_hat_BART-y_test)^2)/length(y_hat_BART)     #0.2876891


## S3 method for class 'bart'
plot(X_train,plquants=c(.05,.95), cols =c('blue','black'))

plot(bart)

extract_raw_node_data(bart_machine,g=5000)


#################################################################################

## ANN

library(neuralnet)
library(GGally)


NN1 <- neuralnet(y_train ~ . , data = train.matrix, hidden = c(4, 1), act.fct = "tanh")

y_hat = predict(NN1, test.matrix, rep = 1, all.units = TRUE)
