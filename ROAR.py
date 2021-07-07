#### Remove and Retrain ####
# This is a version of ROAR (as described in Hooker et al 2019) 
# which works with any scikit-learn model and tabular data (pandas dataframes)




## Libraries ##

#General
import copy
import numpy as np
import pandas as pd

#Evalutation metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

#Interpretation
import shap
shap.initjs()



#### Code ####

## Rankings Function ##
# Trains a model, builds an explanation with KernelShap, and then
# outputs the ranking of each feature in descending order along with
# their shap values.

# Arguments:
# clf : the model to be trained
# X_train: Training data as pandas dataframe
# y_train: Target values for training (This was build using a binary target)
# X_test:  Test data as pandas dataframe

def rankings(clf, X_train, y_train, X_test):
    #Define model
    model = copy.deepcopy(clf)
    
    #Fit model
    model.fit(X_train, y_train)
    
    #Build Explanation
    explainer = shap.KernelExplainer(model.predict_proba, np.mean(X_train))
    shap_values = explainer.shap_values(X_test)
    
    #Get Rankings
    if len(shap_values) == 1:
        values = shap_values
    else: 
        values = shap_values[1]

    shap_values_df = pd.DataFrame(values, columns = X_train.columns)
    rankings = shap_values_df.abs().mean().sort_values(ascending = False)
    
    return rankings

## Retrain Functions ##
# Retrains a model iteratively as we 
# remove the features. It outputs
#the accuracy, balanced accuracy, and 
#f1 scores for each model

# Arguments:
# clf: model to be trained
# X_train: List of dataframes for the training data where features are removed as we iterate through it
# y_train: Target of the training data
# X_test: List of dataframes for the test data where features are removed as we iterate through it
# y_test: Target of the test set. 

#General Retrain function
def retrain(clf, X_train, y_train, X_test, y_test):
    accu = []
    accu_balanced = []
    f1 = []
    #Iterate through the datasets and retrain for each
    for i in range(len(X_train)):
        clf_ret = copy.deepcopy(clf)
        
        #Fit model
        clf_ret.fit(X_train[i], y_train)
        
        #Get metrics: 
        y_pred = clf_ret.predict(X_test)
        y_true = y_test
        
        accu.append(accuracy_score(y_true, y_pred))
        accu_balanced.append(balanced_accuracy_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred))
        
    return accu, accu_balanced, f1




## Remove Features ## 
#These functions create a list of dataframes 
#where they remove a set percentage of the features.
#They remove from the top %, bottom %, or random.

# Arguments:
# t: percentage of features to be removed in each iteration
# rankings: a list of rankings (descending rankings) to guide the removal
# X_train, X_test: dataframes of the data from which we will remove the features


#Remove top t features
def remove_top(t, rankings, X_train, X_test):
    top_t = np.round(len(rankings)*t).astype(int)
    iters = np.ceil(len(rankings)/top_t).astype(int)

    high_rankings = copy.deepcopy(rankings)
    xtr = copy.deepcopy(X_train)
    xts = copy.deepcopy(X_test)


    X_train_ = {}
    X_test_ = {}
    to_remove = []
    removed = []
    counter = 0
    # high_rankings.drop(to_remove)

    for i in range(iters-1):
        to_remove = np.array(eval(f"high_rankings.iloc[0:{top_t+counter}]").index)
        counter += top_t
        to_remove = to_remove[np.isin(to_remove, removed, invert = True)]
        for col in to_remove:
            removed.append(col)
            xtr[col].replace(xtr[col].values, xtr[col].mean(), inplace = True)
            xts[col].replace(xts[col].values, xts[col].mean(), inplace = True)
        X_train_[i] = copy.deepcopy(xtr)
        X_test_[i] = copy.deepcopy(xts)

    return [X_train_, X_test_]



#Remove lowest t features
def remove_bottom(t, rankings, X_train, X_test):
    bottom_t = np.round(len(rankings)*t).astype(int)
    iters = np.ceil(len(rankings)/bottom_t).astype(int)

    bottom_rankings = copy.deepcopy(rankings)
    xtr = copy.deepcopy(X_train)
    xts = copy.deepcopy(X_test)
    
    X_train_ = {}
    X_test_ = {}
    to_remove = []
    removed = []
    counter = 0
    # high_rankings.drop(to_remove)

    for i in range(iters-1):
        to_remove = np.array(eval(f"bottom_rankings.iloc[-{bottom_t+counter}:-1]").index)
        counter += bottom_t
        to_remove = to_remove[np.isin(to_remove, removed, invert = True)]
        for col in to_remove:
            removed.append(col)
            xtr[col].replace(xtr[col].values, xtr[col].mean(), inplace = True)
            xts[col].replace(xts[col].values, xts[col].mean(), inplace = True)
        X_train_[i] = copy.deepcopy(xtr)
        X_test_[i] = copy.deepcopy(xts)
    return [X_train_, X_test_]


#Remove t random features
def remove_random(t, rankings, X_train, X_test): 
    random_t = np.round(len(rankings)*t).astype(int)
    iters = np.ceil(len(rankings)/random_t).astype(int)
    choices = np.random.choice(len(rankings), len(rankings), replace = False)

    random_rankings = copy.deepcopy(rankings)
    xtr = copy.deepcopy(X_train)
    xts = copy.deepcopy(X_test)


    X_train_ = {}
    X_test_ = {}
    to_remove = []
    removed = []
    counter = 0
    # high_rankings.drop(to_remove)

    for i in range(iters-1):
        to_remove = np.array(random_rankings[choices[0:random_t+counter]].index)
        counter += random_t
        to_remove = to_remove[np.isin(to_remove, removed, invert = True)]
        for col in to_remove:
            removed.append(col)
            xtr[col].replace(xtr[col].values, xtr[col].mean(), inplace = True)
            xts[col].replace(xts[col].values, xts[col].mean(), inplace = True)
        X_train_[i] = copy.deepcopy(xtr)
        X_test_[i] = copy.deepcopy(xts)
    return [X_train_, X_test_]



## Remove  Wrapper##
# Wrapper for the "Remove functions" which makes them easier to handle

# Arguments:
# t: percentage of features to be removed in each iteration
# rankings: a list of rankings (descending rankings) to guide the removal
# X_train, X_test: dataframes of the data from which we will remove the features
# type: which remove function to use "top", "bottom", or "random"

def remove(t, rankings, X_train, X_test, type = "top"):
    if type == "top":
        return remove_top(t, rankings, X_train, X_test)
    elif type == "bottom":
        return remove_bottom(t, rankings, X_train, X_test)
    elif type == "random":
        return remove_random(t, rankings, X_train, X_test)
    
    
    
## ROAR ##
# The main function of the library.
#Wraps all other functions in a nice pipeline which is easy to use.
#Accepts any scikit-learn model. It was built and tested using a binary target.

# Arguments:
# clf : the model to be trained
# t: percentage of features to be removed in each iteration
# X_train: Training data as pandas dataframe
# y_train: Target values for training (This was build using a binary target)
# X_test:  Test data as pandas dataframe

#outputs accuracy, balanced_accuracy, f1_score, and ranks for each iteration.
    
def roar(clf, t, X_train, y_train, X_test, y_test):
    accu = {}
    bal_accu = {}
    f1 = {}
    ranks = []
    #Repeat 5 times:
    for i in range(5):
        #Get rankings
        ranks.append(rankings(clf, X_train, y_train, X_test))
        #Repeat 3 times removing from top, bottom, and randomly
        for k in ["top", "bottom", "random"]:
            #Remove
            removed = remove(t, ranks[i], X_train, X_test, type = k)
            #Retrain
            accu[i,k], bal_accu[i,k], f1[i,k] = retrain(clf, removed[0], y_train, X_test, y_test)
    return [accu, bal_accu, f1, ranks, t]


## Plot the output of ROAR ##
def plot_metrics(roar_output):
    step = roar_output[4]*100
    x_axis_breaks = np.arange(step,100, step)
    
    df = pd.DataFrame(roar_output[0]).mean(axis = 1, level = 1)
    df.index =  x_axis_breaks
    df.plot(title = f"ROAR with KernelShap (Accuracy)", xlabel = "% of input features removed", ylabel = "Accuracy", figsize = (10,8), grid = True)
    
    df = pd.DataFrame(roar_output[1]).mean(axis = 1, level = 1)
    df.index =  df.index =  x_axis_breaks
    df.plot(title = f"ROAR with KernelShap (Balanced Accuracy)", xlabel = "% of input features removed", ylabel = "balanced_accuracy", figsize = (10,8), grid = True)
    
    df = pd.DataFrame(roar_output[2]).mean(axis = 1, level = 1)
    df.index =  x_axis_breaks
    df.plot(title = f"ROAR with KernelShap (F1_Score)", xlabel = "% of input features removed", ylabel = "f1_score", figsize = (10,8), grid = True)