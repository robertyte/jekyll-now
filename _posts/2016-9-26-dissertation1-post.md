---
layout: post
title: DISS-1. About goals, complexities and approach
tag: []
---

**My dissertation** investigates machine learning approaches to detect rare hereditary diseases retrospectively studying large population cohorts. Rare diseases are diseases which affect a small number of people (1 out of 2,000). Specific issues are raised in relation to their rarity: many patients are not diagnosed, the diagnosis is delayed or wrongly determined resulting in inappropriate treatments. Therefore a tremendous need to investigate novel machine learning tools exists to enable early and more accurate rare disease detection. 

I have analysed life-threatening genetic condition which occurs in about 1 in 10,000 to 1 in 150,000 people worldwide (for confidentiality reasons I will not disclose the name of the disease, instead hereinafter, it will be called 'the condition'). My study shows, for the first time in the literature, the use of machine learning techniques aimed to capture early flags of the condition to provide clinical decision support for clinicians. 

Analysed data set comprised of 165 million patients and possesed severe factual imbalance of cases to controls (1:138,000). Data encompassed medical history events which were described by 240 variables.

The research was divided into following main parts: * Exploratory data analysis, * Experimental Phases I. Supervised learning classifiers, * Experimental Phases II. Tackling imbalance, * Experimental Phases III. Classifiers’ ensembles, * Feasibility assessment of predictive model’s deployment. 

Six supervised learning algorithms - logistic regression, SVM with RBF kernel, Decision Tree, Random Forest, AdaBoost, and Naïve Bayes - and unsupervised One Class SVM were tested. As well PCA and recursive feature elimination (RFE) techniques were used to reduce dimensionality and explore feature importance. For tackling imbalance Parallel Selective Sampling (PSS) method was reproduced in Apache Spark implementation. Finally, the majority voting rule for models ensemble and unique cascade ensemble were explored.




