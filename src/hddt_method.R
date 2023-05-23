# Funções necessárias para a árvore de decisão ----

# build a Hellinger distance decision tree
# it is a recursive function that calls itself with subsets
# of training data that matches the decision criterion
# using a list to create the tree structure
#
# Input
# X (matrix/data frame): training data, features/independent variables
#                       The columns of X must be either numeric or factor
# y (vector)           : training data, labels/dependent variable
# C (integer)          : minimum size of the training set at a node to attempt a split
# labels (vector)      : allowed labels [optional]
#
# Value
# node (list)          : the root node of the deicison tree
HDDT <- function(X, y, C, labels=unique(y)) {
  
  if(is.null(labels) || length(labels)==0) labels <- unique(y)  
  
  node <- list() # when called for first time, this will be the root
  node$C <- C
  node$labels <- labels
  print(labels)
  print('Checkpoint 1')
  print(unique(y))
  if(length(unique(y))==1 || length(y) < C) {
    print('Checkpoint 2')
    # calculate counts and frequencies
    # use Laplace smoothing, by adding 1 to count of each label
    y <- c(y, labels)
    node$count <- sort(table(y), decreasing=TRUE)
    node$freq  <- node$count/sum(node$count)
    # get the label of this leaf node
    node$label <- as.integer(names(node$count)[1])
    
    return(node)
  }
  else { # recursion
    # get Hellinger distance and their max
    # use for loop insread of apply as it will convert data.frame to a matrix and mess up column classes
    # e.g. factor will get coerced into character
    print('Checkpoint 3')
    HD <- list()
    for(i in 1:ncol(X)) HD[[i]] <- HDDT_dist(X[,i],y=y,labels=labels) 
    print('Checkpoint 3.1')
    hd <- sapply(HD, function(x) {return(x$d)})
    i  <- which(hd==max(hd))[1] # just taking the first 
    print('Checkpoint 3.2')
    # save node attributes
    node$i    <- i
    node$v    <- HD[[i]]$v
    node$type <- HD[[i]]$type
    node$d    <- HD[[i]]$d
    
    if(node$type=="factor") {
      print('Checkpoint 4')
      j <- X[,i]==node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)
    }
    else if(node$type=="numeric") {
      print('Checkpoint 5')
      j <- X[,i]<=node$v
      node$childLeft  <- HDDT(X[j,], y[j], C, labels)
      node$childRight <- HDDT(X[!j,], y[!j], C, labels)      
    }
  }
  
  return(node) # returns root node
}

# given the root node as returned by the HDDT function and
# new data X return predictions
#
# Input
# root (list)           : root node as returned by the function HDDT
# X (matrix/data frame) : new data, features/independent variables
#
# Value
# y (integer vector)    : predicted labels for X
HDDT_predict <- function(root, X) {
  y <- rep(NA, nrow(X))
  for(i in 1:nrow(X)) {
    # traverse the tree until we find a leaf node
    node <- root
    while(!is.null(node$v)) {
      if(node$type=="factor") {
        if(X[i,node$i]==node$v) node <- node$childLeft
        else node <- node$childRight
      }
      else if(node$type=="numeric") {
        if(X[i,node$i]<=node$v) node <- node$childLeft
        else node <- node$childRight
      }
      else stop("unknown node type: ", node$type)
    }
    stopifnot(!is.null(node$label))
    y[i] <- node$label
  }
  
  return(y)
}


# given a feature vector calculate Hellinger distance
# it takes care of both discrete and continuous attributes
# also returns the "value" of the feature that is used as decision criterion
# and the "type" pf the feature which is either factor as numeric
# ONLY WORKS WITH BINARY LABELS
HDDT_dist <- function(f, y, labels=unique(y)) {
  print('Checkpoint erro aqui')
  print(labels[1])
  print(labels[2])
  i1 <- y==labels[1]
  i0 <- y==labels[2]
  T1 <- sum(i1)
  T0 <- sum(i0)
  val <- NA
  hellinger <- -1
  
  cl <- class(f)  
  if(cl=="factor") {    
    for(v in levels(f)) {
      Tfv1 <- sum(i1 & f==v)
      Tfv0 <- sum(i0 & f==v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else if(cl=="numeric") {
    fs <- sort(unique(f))
    for(v in fs) {
      Tfv1 <- sum(i1 & f<=v)
      Tfv0 <- sum(i0 & f<=v)
      
      Tfw1 <- T1 - Tfv1
      Tfw0 <- T0 - Tfv0
      cur_value <- ( sqrt(Tfv1 / T1) - sqrt(Tfv0 / T0) )^2 + ( sqrt(Tfw1 / T1) - sqrt(Tfw0 / T0) )^2
      
      if(cur_value > hellinger) {
        hellinger <- cur_value
        val <- v
      }
    }
  }
  else stop("unknown class: ", cl)
  
  return(list(d=sqrt(hellinger), v=val, type=cl))
}

# Importação da base ----
library(tidyverse)
library(pROC)
library(caret)
install.packages("Metrics")
library(Metrics)



base <- read.csv('creditcard.csv')
base %>% head()
base <- base %>% head(100000)
temp <- HDDT(X = base %>% select(-Class), y = base[, 'Class'], C = 10, labels = c(0,1))




hddt_tree <- function(X_train, y_train, X_test, y_test){
  f_score_list <- c()
  size_list <- c(5, 50, 100,250, 500, 1000, 3000, 5000, 7000, 10000, 50000, 100000)
  i <- 1
  for(size in size_list){
    tree <- HDDT(X = X_train, y = y_train, C = size)
    vetor_predict  <- HDDT_predict(tree, X_test)
    
    rec <- recall(y_test,vetor_predict)
    prec <- precision(y_test,vetor_predict)
    
    f_score_calc <- (2 * (prec * rec)) / (prec + rec) 
    print(f_score_calc)
    
    f_score_list[i] <- f_score_calc
    i <- i + 1
  }
  return(list(f_score_list, size_list))
}

sample <- sample(c(TRUE, FALSE), nrow(base), replace=TRUE, prob=c(0.7,0.3))
train  <- base[sample, ]
test   <- base[!sample, ]

X_train <- train %>% select(-Class)
X_test <- test %>% select(-Class)

y_train <- train[, 'Class']
y_test <- test[, 'Class']
tuple_out <- hddt_tree(X_train, y_train, X_test, y_test)
f1s <- tuple_out[[1]]
sizes <- tuple_out[[2]]


