---
layout: post
title: DISS-1. About goals, approach and complexities 
tag: []
---

In the dissertation I investigated machine learning approaches to detect rare hereditary diseases retrospectively studying large population cohorts. 

Rare diseases are diseases which affect a small number of people (1 out of 2,000). Specific issues are raised in relation to their rarity: many patients are not diagnosed, the diagnosis is delayed or wrongly determined resulting in inappropriate treatments. Therefore a tremendous need to investigate novel machine learning tools exists to enable early and more accurate rare disease detection. 

I analysed life-threatening genetic condition which occurs in about 1 in 10,000 to 1 in 150,000 people worldwide (_for confidentiality reasons I will not disclose the name of the disease, instead hereinafter, I will call it 'the condition'_). My study shows, for the first time in the literature, the use of machine learning techniques aimed to capture early flags of the condition to provide clinical decision support for clinicians. 

Analysed data set comprised of 165 million patients and possesed severe factual imbalance of cases to controls (1:138,000). Data encompassed medical history events which were described by 240 variables.

I structured my methodological approach into following distinct parts: 

  - **Exploratory data analysis** with frequency and PCA plots, 
  
  - **Experimental Phases I. Testing six single supervised learning classifiers** - logistic regression, SVM with RBF kernel, Decision Tree, Random Forest, AdaBoost, and Naïve Bayes. Feature elimination and signal dilution eith varying cases vs. controls ratio in hold out data set, 
  
  - **Experimental Phases II. Tackling imbalance** with One Class SVM and Parallel Selective Sampling in Apache PySpark implementation, 
  
  - **Experimental Phases III. Classifiers’ ensembles**: majority vote and cascade ensemble, 
  
  - **Feasibility assessment** of model’s deployment. 


The results of experiments revealed two key complexities:

  1. **Severe factual data set imbalance** of positive vs. negative classes, which required adoption of undersampling techniques.
  
  2. **Positive class heterogeneity**. Firstly, the disorder itself per se is not homogeneous - there are several types of this dissorder and many genetic mutations responsible for it. Secondly, the heterogeneity was reinforced within analysed feature space.

In the later posts I will disclose some portions of the code and more details on the methodology.


