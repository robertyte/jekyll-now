---
layout: post
title: DISS-5. One Class SVM
tag: []
---

When your Two Class learning process suffers from heterogeneity in the positive class and severe class imbalance, One Class SVM can come in handy.

One Class SVM is an unsupervised algorithm that learns a decision function for novelty (or anomalies) detection: classifying new data as similar or different to the training set. By just providing training data of one class, an algorithm creates a representational model of this data. If newly encountered data is too different, according to some measurement, from this model, it is labelled as out-of-class. Very nice explanation of mechanics and parameters of One CLass SVM could be found in [Scholkopf, B. et al. (2000) Support Vector Method for Novelty Detection. MIT Press (2000)].

I used One Class SVM with RBF kernel to tackle two issues within the data set of dissertation. Firstly, positive class possessed high heterogeneity, thus I assumed learning the core cluster of positive cases and omitting the outlying instances could improve performance. Secondly, there was a severe imbalance of 1:138.000 positive vs. negative instances within data set, thus any conventional two class models becomes incapable. And since one Class SVM model learns only on the minority class the imbalance and big data issue becomes obsolete.

The results revealed that One Class SVM did not do a good job at my specific data set. At the end too many of test instances were classified as positive and thus the sensitivity went up but specificity plunged. It could indicate that the heterogeneity of positive group could not be tackled and so model learned on wide spectrum of positive cases, which afterwards impeded classifier’s capability to distinguish many negative instances from positive ones.

But that is not always the case, the success (or failure) of One Class SVM is data set dependent. And it cna be measured empirically. 

One Class SVM with RBF kernel has two parameters - gamma and nu - which are empirically set. Nevertheless, because of distinct nature of the learning setting standard cross-validated grid search methods from _sklearn_ cannot be used to find the best fitting parameters. Thus I provide code stubs 've written myself to empirically find the best parameters and train the One Class SVM.


```python
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn import cross_validation

def OneClassSVM_gridSearch(positive_train, negative_train, n_folds, n):
    '''
    This function returns trained model with best parameters of gamma for RBF kernel and nu.
    positive_train: pandas dataframe - positive samples with independent variables for grid search. Normalized if needed.
    negative_train: pandas dataframe - negative samples with independent variables for grid search. Normalized if needed. negative_train dataframe must be of the same size (rows and columns) as positive_train.
    n_folds: integer - number of folds in grid search.
    n: integer - index of iteration.
    
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    # set a range of possible parameters' values for grid searching.
    gamma=[0.00000001,0.0000001,0.000001,0.00001,0.0001, 0.001, 0.01]
    nu=[0.00001, 0.0001, 0.001, 0.01]
    
    columns = ['iterationNo','gamma', 'nu', 'avgScore']
    exp = pd.DataFrame(columns=columns)
    # split the data into n_folds for cross-validation
    kf = cross_validation.KFold(positive_train.shape[0], n_folds=n_folds)
    for i, element in enumerate(itertools.product(gamma, nu)):
        clf.set_params(gamma = element[0], nu = element[1])
        score = 0.0
        for kf_train, kf_test in kf:
            # train model only on the positive class train folds
            clf.fit(positive_train.iloc[kf_train])
            # test model both on positive and negative classes test fold
            X_test = positive_train.iloc[kf_test].append(negative_train.iloc[kf_test])
            y_test = [1]*len(kf_test) + [-1]*len(kf_test)
            score += metrics.accuracy_score(y_test, clf.predict(X_test))
        # calculate average of all accuracy scores from test folds and store it in dataframe for later analysis
        avgScore = score/n_folds
        exp.loc[i,'iterationNo'] = n
        exp.loc[i,'gamma'] = element[0]
        exp.loc[i,'nu'] = element[1]
        exp.loc[i,'avgScore'] = avgScore
    # get parameters pair with the highest average score
    best_gamma = exp.ix[exp['avgScore'].idxmax()]['gamma']
    best_nu = exp.ix[exp['avgScore'].idxmax()]['nu']
    # train One Class SVM with best parameters on full train set
    clf.set_params(gamma = best_gamma, nu = best_nu)
    return clf.fit(positive_train)

# use the above defined function to run 50 iterations of training and evaluating One Class SVM
# positive: dataframe with all positive class instances and their dependent variables
# negative: dataframe with negative class instances (>= # of positive class instance) and their dependent variables
iterations = 50
rs = cross_validation.ShuffleSplit(positive.shape[0], n_iter=iterations, test_size=.25, random_state=0)

OCSVM = pd.DataFrame(columns=['Sensitivity', 'Sepcificity', 'g-mean', 'Precision'])
for n, (train_index, test_index) in enumerate(rs):
    # prepare training data
    positive_train = positive.iloc[train_index] 
    negative_train = negative.iloc[train_index]
    #train the model
    n_folds = 4
    clf = OneClassSVM_gridSearch_Score(positive_train, negative_train, n_folds, n)
    # test and evaluate the model
    y_predict = clf.predict(positive.iloc[test_index].append(negative.iloc[test_index]))
    y_true = [1]*len(test_index)+[-1]*len(test_index)
    sensitivity= metrics.recall_score(y_true, y_predict)
    specificity = 1.0-metrics.roc_curve(y_true,y_predict)[0][1]
    #save the results for further analysis
    OCSVM.loc[n,'Sensitivity'] = sensitivity
    OCSVM.loc[n,'Sepcificity'] = specificity
    OCSVM.loc[n,'g-mean'] = np.sqrt(sensitivity*specificity)
    OCSVM.loc[n,'Precision'] = metrics.precision_score(y_true, y_predict)
    print "Done with iteration no.: ", n
```

