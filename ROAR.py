#### Remove and Retrain ####
# This is a version of ROAR (as described in Hooker et al 2019) 
# which works with any scikit-learn model and tabular data (pandas dataframes)




## Libraries ##
#General
import copy
import numpy as np
import pandas as pd

#Plotting
import matplotlib as plt

#Evalutation metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

#Interpretation
import shap


#Keeping track of time
from fastprogress.fastprogress import progress_bar



#### Code ####

## Explanation Function ##
# Trains a model, builds an explanation with KernelShap, and then
# outputs the ranking of each feature in descending order.
# Arguments:
# clf : A trained ML model with a .predict method
# X: Training data as pandas dataframe
# x:  Test data as pandas dataframe
# explainer: any explainer that follows the shap api format
def explain(clf, X, x, explainer = shap.explainers.Permutation):
    if explainer != shap.explainers.Tree:
        # #Build Explanation
        explanation = explainer(clf.predict, X)
        shap_values = explanation(x)
    else:
        # #Build Explanation
        explanation = explainer(clf, X)
        shap_values = explanation(x, check_additivity = False)
    #Get SHAP values for positive class
    shap_values = shap_values.values[...,1]
    shap_values = np.abs(shap_values)
    shap_values = np.mean(shap_values, axis= 0)
    #Get ranks
    ranks = np.argsort(shap_values)
    ranks
    return ranks


## Training Function ##
# Utility function to train the a copy of a model.
# Arguments:
# clf: model to be trained
# X: a pandas dataframe containing training data
# Y: Target of the training data (pd.DataFrame)

#General Retrain function
def train(clf, X, Y):
    accu = np.array([])
    accu_balanced = np.array([])
    f1 = np.array([])
    #Iterate through the datasets and retrain for each
    temp = copy.deepcopy(clf)

    #Fit model
    temp.fit(X, Y)
    
    return temp


## Metrics function ##
# Utility function which outputs a series of metrics to evaluate
# Currently gets accuracy, balanced_accuracy, and f-score

#Arguments:
#clf: A trained ML model with a predict method
#x: a pd.DataFrame of test data
#y: Target of the test data
def metrics(clf, x, y):   
    #Get metrics: 
    yhat = clf.predict(x)

    accu = accuracy_score(y, yhat)
    accu_balanced = balanced_accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
        
    return np.array([[accu], [accu_balanced], [f1]])




## Mask Features ## 
#These functions mask data and retrain the ML model
#They mask from the top %, bottom %, or random.

# Arguments:
# t: percentage of features to be masked in each iteration
# rankings: a list of rankings (descending rankings) to guide the removal
# X, x: dataframes of the data from which we will remove the features. Training set and testing set, respectively
# Y, y: Targets for the training and testing sets, respectively
#clf: Model to retrain
# base: metrics of the full model


#Remove top t features
def remove_top(t, rankings, X, Y, x, y, clf, base):
    
    #Make copies of our data to modify
    X_train = copy.deepcopy(X)
    x_test = copy.deepcopy(x)
    results = copy.deepcopy(base)

    #Set masking schedule
    j = int(np.round(len(rankings)*t))
    i = 0
    k = j
    
    #Mask and retrain
    while k <= len(rankings)+1:
        #Mask
        X_train.iloc[:, rankings[i:k]] = X_train.iloc[:, rankings[i:k]].mean()
        x_test.iloc[:, rankings[i:k]] = x_test.iloc[:, rankings[i:k]].mean() 
        #Retrain
        model = train(clf, X_train, Y)
        results =  np.hstack((results, metrics(model, x_test, y)))
        
        #Move iterator forward
        i += j
        k += j

    return results



#Remove lowest t features
def remove_bottom(t, rankings, X, Y, x, y, clf, base):
    #Make copies of our data to modify
    X_train = copy.deepcopy(X)
    x_test = copy.deepcopy(x)
    results = copy.deepcopy(base)
    
    
    #Set masking schedule
    j = int(np.round(len(rankings)*t))
    i = len(rankings) - j
    k = len(rankings)
    
    #Mask and retrain
    while k >= j:
        #Mask
        X_train.iloc[:, rankings[i:k]] = X_train.iloc[:, rankings[i:k]].mean()
        x_test.iloc[:, rankings[i:k]] = x_test.iloc[:, rankings[i:k]].mean() 
            
        #Retrain
        model = train(clf, X_train, Y)
        results =  np.hstack((results, metrics(model, x_test, y)))
        
        
        #Move iterator forward
        i -= j
        k -= j

    return results


#Remove t random features
def remove_random(t, rankings, X, Y, x, y, clf, base):
    
    random_choices = np.random.choice(len(rankings), len(rankings), replace = False)

    #Make copies of our data to modify
    X_train = copy.deepcopy(X)
    x_test = copy.deepcopy(x)
    results = copy.deepcopy(base)
    
    #Set masking schedule
    j = int(np.round(len(random_choices)*t))
    i = 0
    k = j
    
    #Mask and retrain
    while k <= len(random_choices):
        #Mask
        X_train.iloc[:, random_choices[i:k]] = X_train.iloc[:, random_choices[i:k]].mean()
        x_test.iloc[:, random_choices[i:k]] = x_test.iloc[:, random_choices[i:k]].mean() 
            
        #Retrain
        model = train(clf, X_train, Y)
        results = np.hstack((results, metrics(model, x_test, y)))


        #Move iterator forward
        i += j
        k += j

    return results
    
    
    
## ROAR ##
# The main function of the library.
#Wraps all other functions in a nice pipeline which is easy to use.
#Accepts any scikit-learn model. It was built and tested using a binary target.

# Arguments:
# clf : the model to be trained
# t: percentage of features to be removed in each iteration
# X: Training data as pandas dataframe
# Y: Target values for training (This was build using a binary target)
# x:  Test data as pandas dataframe
# y: Target values for testing
# explainer: any explainer which built with the shap api
# repeats: how many times to explain and do the whole retraining

#outputs accuracy, balanced_accuracy, f1_score, and ranks for each iteration.  
def roar(X, Y, x, y, clf, explainer = shap.explainers.Permutation, t = 0.10, repeats = 10):
    #Initialize the frames
    model = train(clf, X, Y)
    base = metrics(model, x, y)
    #Initialize
    ranks = explain(model, X, x, explainer)
    top = remove_top(t, ranks, X, Y, x, y, clf, base)
    bottom = remove_bottom(t, ranks, X, Y, x, y, clf, base)
    random = remove_bottom(t, ranks, X, Y, x, y, clf, base)

    #Set progress bar
    mb = progress_bar(range(repeats - 1))
    #Repeat x times
    for i in mb:
        ranks = explain(model, X, x, explainer)
        top = np.dstack(top, remove_top(t, ranks, X, Y, x, y, clf, base))
        bottom = np.dstack(bottom, remove_bottom(t, ranks, X, Y, x, y, clf, base))
        random = np.dstack(random, remove_bottom(t, ranks, X, Y, x, y, clf, base))
    return np.array([top, bottom, random])


## plot the outputs of ROAR ##
#Plot the decay measured in accuracy
def plot_curves(results, explainer = 'KernelShap'):
    step = roar_output[4]*100
    x_axis_breaks = np.arange(step,100, step)
    
    df = pd.DataFrame(roar_output[0]).mean(axis = 1, level = 1)
    df.index =  x_axis_breaks
    df.plot(title = f"ROAR with {explainer} (Accuracy)", xlabel = "% of input features removed", ylabel = "Accuracy", figsize = (10,8), grid = True)
    
    df = pd.DataFrame(roar_output[1]).mean(axis = 1, level = 1)
    df.index =  df.index =  x_axis_breaks
    df.plot(title = f"ROAR with {explainer} (Balanced Accuracy)", xlabel = "% of input features removed", ylabel = "balanced_accuracy", figsize = (10,8), grid = True)
    
    df = pd.DataFrame(roar_output[2]).mean(axis = 1, level = 1)
    df.index =  x_axis_breaks
    df.plot(title = f"ROAR with {explainer} (F1_Score)", xlabel = "% of input features removed", ylabel = "f1_score", figsize = (10,8), grid = True)