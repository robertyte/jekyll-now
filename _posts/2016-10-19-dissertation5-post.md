---
layout: post
title: DISS-5. One Class SVM
tag: []
---

When your Two Class learning process suffers from heterogeneity in the positive class and severe class imbalance, One Class SVM can come in handy.

To start from the aim, I was seeking to understand which variables are redundant, correlated and thus introduce noise into the models, and which variables are the most deterministic. Thus, I was expecting that elimination of features would improve the overall performance or, at least, reduce the computational complexity by reducing dimensionality (I had 240 variables at the beginning). 

To conduct the experiment I used the following function. It is important to note that only those estimators can be used within RFECV which have _coef_ as an attribute available, e.g. SVM with RBF kernel and Naive Bayes classifier could not be used, thus I ran experiments only with logistic regression, decision tree, random forest and AdaBoost estimators. 
```python
from sklearn.feature_selection import RFECV
```

With each of four classifiers I repeated RFECV for 50 runs and averaged outcomes. The results were as follows: in average logistic regression retained 230 out of 240 variables, decision tree - 75, random forest - 73, AdaBoost - 33. The decrease of avg. AUC in most of the cases was marginal - logistic regression, random forest and AdaBoost detoriorated only by 2%, decision tree - by 12%. This confirms a major improvement in computational efficiency of the task (especially in the tree based classifiers) and redundancy of many features. 

For further analysis I plot features on x-axis and on y-axis the number of times (out of 50) when feature was chosen for inclusion in the final reduced model. (below for simplicity reasons I depict only a fraction of 78 features out of all 240)

![ROC Space curve](../images/CVRFE_analysisOfFeats.png)

It is quite clear that some features are more dispersed than the others. E.g. features 6, 71-74 are unanimously selected by all 4 models in majority of the runs. Contrary, all models cannot agree on 53, 67 features. To better visualize this disagreement I further exclude logistic regression from the analysis, average 'times selected' of remaining three models - decision tree, random forest and AdaBoost - and add error bars, which results in following plot.  

![ROC Space curve](../images/CVRFE_analysisOfFeats_onlyTrees.png)

The longer the whiskers of error bars the more disagreement between three classifiers, while short whiskers and high position of dots indicate big importance of the feature across all models.

As always, below are provided code stubs for reproducing similarexperiment and the plots.


```python
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn import cross_validation

from sklearn.feature_selection import RFECV

def CVRFE(X, y, iterations):
    '''
    This function runs CVRFE multiple times with four classifiers and for each classifier per each run records results of CVRFE in a dataframe exp.
    X: pandas dataframe - data set with independent variables.
    y: list,array - dependent variable
    iterations: integer - number of random train/test splits
    '''
    names = [ 
            "Logistic_regression", 
            "Decision_Tree",
            "Random_Forest", 
            "AdaBoost" 
            ]
    classifiers = [
        LogisticRegression(penalty = 'l2'),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=6),
        AdaBoostClassifier(n_estimators=10)
        ]
    
    columns = [[name+'_selectedFeatures', name+'_auc',  name+'_sensitivity', name+'_specificity'] for name in names]
    columns = sum(columns,[])
    exp = pd.DataFrame(columns=columns)
    rs = cross_validation.ShuffleSplit(len(y), n_iter=iterations, test_size=.25, random_state=0)
    i =0
    y = np.array(y)
    for train_index, test_index in rs:
        for name, clf in zip(names, classifiers):
                # prepare data
                X_train = X.iloc[train_index] 
                y_train = y[train_index]
                X_test = X.iloc[test_index]
                y_test = y[test_index]
                # run RFECV on training data
                clf_GS = RFECV(clf, step=1, cv=4, scoring='roc_auc', n_jobs = 6)
                clf_GS.fit(X_train, y_train)
                selectedFeat = clf_GS.support_
                exp.loc[i,name+'_selectedFeatures'] = selectedFeat
                # train model on reduced train data set 
                clf.fit(X_train.loc[:,selectedFeat], y_train)
                # test model on reduced test data set
                pred_prob_class1 = clf.predict_proba(X_test.loc[:,selectedFeat])[:,1]
                pred_label = clf.predict(X_test.loc[:,selectedFeat])
                auc = metrics.roc_auc_score(y_test, pred_prob_class1)
                
                exp.loc[i,name+'_auc']=auc
                exp.loc[i,name+'_sensitivity']= metrics.recall_score(y_test,pred_label)
                exp.loc[i,name+'_specificity']= 1.0-metrics.roc_curve(y_test,pred_label)[0][1]
        i +=1        
    return exp
```

