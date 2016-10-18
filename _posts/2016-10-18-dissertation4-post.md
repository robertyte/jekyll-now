---
layout: post
title: DISS-4. Feature importance
tag: []
---

And now about my creative approach to analyse feature importance with cross-validated recursive feature elimination method from sklearn.



![ROC Space curve](../images/CVRFE_analysisOfFeats.png)

In addition to averaged AUC, specificity and sensitivity as a performance diagnostics I plotted averaged ROC Space curve. It represents averaged sensitivity and specificity metrics over number of iterations on ROC plot with 95% confidence interval error bars.
I used simplified version of ROC Space curve with only one point depicted. Nevertheless, more sophisticated versions of averaging ROC Space curves exist - merging all iterations into one curve, vertical averaging, threshold averaging <sup>1</sup>.

![ROC Space curve](../images/CVRFE_analysisOfFeats_onlyTrees.png)

The interpretation of such curve is straight forward - the most desirable performance outcome would result in points placed on the furthest upper left corner with narrow confidence intervals. In our case the error bars are sufficiently narrow confirming stable performance of classifiers per varying train/test splits, though the true positive rates (or sensitivity) are too low.

Below I share code stubs for plotting averaged ROC Space curve, similar to one above in the image.


```python
def ROCSpace_1point(pred_truthLabels, model):
    '''
    This function takes as input predefined dataframe with results of experimental runs for one classifier 
    and returns a plot of averaged ROC space curve.
    pred_truthLabels: dataframe - for one classifier containing 2 columns: 'Truth' with lists of true labels and 
    'PredLabels' with lists of predicted labels from test subsets. Each row represents different iteration of experiment, 
    eg. if experiment was run 100 times there should be 100 rows in the dataframe.
    model: string - name of classifier.
    '''

    plt.figure(figsize=[10,10])
    
    pred_truthLabels['Sensitivity'] = pred_truthLabels.apply(lambda x: metrics.recall_score(x['Truth'],x['PredLabels']), axis=1)
    pred_truthLabels['Specificity'] = pred_truthLabels.apply(lambda x: 1.0-metrics.roc_curve(x['Truth'],x['PredLabels'])[0][1], axis=1)
    xerror= 1.96*(1.0-pred_truthLabels['Specificity']).std()/np.sqrt(pred_truthLabels.shape[0])
    yerror = 1.96*pred_truthLabels['Sensitivity'].std()/np.sqrt(pred_truthLabels.shape[0])
    xmean = (1.0-pred_truthLabels['Specificity']).mean()
    ymean = pred_truthLabels['Sensitivity'].mean()
    
    plt.errorbar([0,xmean,1], [0,ymean,1], 
                 yerr=[[0,yerror,0], [0,yerror,0]], xerr=[[0,xerror,0], [0,xerror,0]], 
                 linestyle = ':',elinewidth=2,linewidth = 1, label =model)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.show()
```
<sup>1</sup> Fawcett, T. (2005) "An introduction to ROC analysis". Pattern Recognition Letters 27 (2006) 861–874.
