---
layout: post
title: DISS-6. Parallel Selective Sampling in Apache Spark
tag: []
---

More conventional methods to tackle class imbalance are either under-sampling of majority class, over-sampling of minority class or hybrid. Since the data I analysed was severely imbalanced the best shot was to try an under-sampling technique. The one I came across is called Parallel Selective Sampling. It is specifically designed for Big Data and high imbalance scenarios. 

Many real-word applications of machine learning classiﬁers have to identify rare events from very large data sets. Consequently two problematic issues add on: the computational complexity dependent on the size of the data set and the need to pursue a fairly high rate of correct detections in the minority class. Due to these issues many classiﬁcation algorithms present great limitations on large data sets and show a performance degradation due to class imbalance. One of the ways to overcome these limitations is a selection of examples by sampling a small number of patterns from the majority class to reduce both the number of data and the imbalance. Such a procedure is well known as “under-sampling” method. Several under-sampling methods are presented in literature. In this project I have chosen to experiment with one, named Parallel Selective Sampling (PSS), that is specifically adopted for both big data and severe imbalance challenges. It was developed by co-authors Annarita D’Addabbo and Rosalia Maglietta, and published in 2015. PSS is a ﬁlter method which can be combined with a variety of classiﬁcation strategies. It is based on the idea (usually used in SVM) that only training data, near the separating boundary (for classiﬁcation), are relevant. In this way the core information from the training data - i.e. the training data points near the separating boundary - is preserved while the size of the training set is effectively decreased. Relevant examples from the majority class are selected and used in the successive classiﬁcation step with desired classifier. Due to the complex computational requirements, PSS is conceived and designed for parallel and distributed computing <sup>1</sup>.

PSS is based on the computation of Tomek links [45], deﬁned as a pair of nearest neighbours of opposite classes. Given ${E_{1},E_{2},…E_{N}}∈R^d$, a pair 〖(E〗_i,E_j) is called a Tomek link if E_i and E_j have different labels, and there is not an E_l such that 〖d(E〗_i,E_l)<d(E_i,E_j) or 〖d(E〗_j,E_l)<d(E_i,E_j), where d( •, •) is the Euclidean distance. Here Tomek links are used to remove samples of majority class staying in areas of input space dense of data belonging to the same class.
Let S= {〖(x〗_1,y_1 ),(x_(2,),y_2 ),…〖(x〗_N,y_N)} be the training set, where x_i∈R^d  and y_i∈{0,1},∀ i=1,…,N. We deﬁne S_0 the set of N_0 training data belonging to class y=0 and S_1 the set of N_1  training data belonging to class y=1, with N_0≫N_1. PSS achieves a reduced training set whose percentage M% of the minority class on the total number of examples is chosen by the user.


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
    '''
   
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

<sup>1</sup> Maglietta, R. et al. (2015) Parallel selective sampling method for imbalanced and large data classiﬁcation. Elsevier, vol. 62, 1 September 2015, p. 61–67.
