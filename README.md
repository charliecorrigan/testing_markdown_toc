# Machine Learning at Kapost

# Table of Contents
1. [Helpful Resources](#helpful-resources)
2. [Example2](#example2)
3. [Third Example](#third-example)


Table of Contents: 
Helpful Resources
 Machine Learning in a Nutshell 
Before Getting Started
The Supervised Learning Pipeline
Supervised Learning Algorithms
Glossary

	
## Helpful Resources

Machine Learning Basics
Understanding Machine Learning Presentation: Google Engineer Mind Dump
Machine Learning Video Tutorial Series: Practical Machine Learning Tutorial with Python Intro
Introduction to Machine Learning with Python:  A Guide for Data Scientists - Muller & Guido (O’Reilly, 2017)
	Internal Resources
ML Repo
Research Summary: Predictive Tagging
Project Brief: Machine Learning 2.0
Machine Learning Applications
FY19 Q1 Project Sizing: ML Tagging Improvements
	Further Reading
Amazon Machine Learning Developer Guide
Distil Networks Blog: Building a Streaming Log Processing and Machine Learning System at Distil
Spotify Labs: Commoditizing Music Machine Learning : Services
Hacker News: Machine Learning Discussion
Facebook Recruiting Keyword Extraction Challenge Discussion


Machine Learning in a Nutshell

The basic idea behind machine learning is that, by providing a program with enough data points, that program can extract categorization rules ('learn') from those data points.  Ideally, those rules can then be used to categorize new, unknown data points with a reasonable expectation of accuracy.  

This differs from traditional programming in that the rules of data categorization are divined by the machine itself, rather than by a human expert.  Essentially, machine learning assigns a program the following task:  

'Given a set of data, discover the function that best describes this data'.


At Kapost, we are currently focused on supervised learning, as opposed to unsupervised learning or reinforcement learning.
Before Getting Started

A prerequisite to designing a machine learning pipeline is understanding the dataset and how it relates to the task at hand.

Define what questions you hope to answer with your data.  What is the best way to phrase the question as a ML problem? It's important to keep in mind that ML does not generate new information, nor does it prove causation. It uses pattern recognition to generate predictions about future data based on historical data.  

Assess the quality of your existing data.  How much data do you need to get started?   Are you currently collecting all of the data you will need to featurize your model?  Is the data clean enough for your chosen model, and if not what sort of preprocessing will it require?

Model choice and feature extraction.  There are many models to choose from, and some are more well suited to certain types of data than others.  Models learn to describe the training data based on the features they have been provided with, so choosing an illustrative set of features can heavily impact the quality of the final model.

Logistics.  Machine learning projects can become complex and resource-heavy.  How do you expect the pipeline to scale?  Do you have enough resources allocated to handle the projected scale? 

Outcomes.  What will measure success?



The Supervised Learning Pipeline
- Cleaning your data
- Modeling your data
- Interpreting your results






Supervised Learning Algorithms

K-Nearest Neighbors:  
The simplest machine learning algorithm:  building its model consists only of storing its dataset.  A ‘prediction’ means identifying the nearest data point to the new sample (or the average of k-nearest data points).  This model is a good choice for beginners, but has poor performance on large datasets, high dimensional models, and sparse datasets.  
 

Linear Models: 
Linear machine learning algorithms discover the best line, plane or hyperplane to describe a dataset.  At low dimensions, linear models are easy to visualize using a coordinate system.  At dimensions above three, they become harder to understand but more descriptive of their data. 

As with many ML algorithms, there are two major variants: Regression and Classification.  

Models of linear regression will attempt to predict a numeric value for a given sample by minimizing its distance to the calculated line/plane.  Examples of these algorithms are Ordinary Least Squares, Ridge Regression, and Lasso Regression.

Models of linear classification will attempt to predict a class label for a given sample using the calculated line/plane as its decision boundary.  Examples of these algorithms are Logistic Regression, Linear Support Vector Machines (SVMs), and One vs. Rest.

The equation for linear regression is loosely based on the equation for a line (y = mx + b, where m = slope and b = y-intercept).  In linear regression, ‘y’ is the final prediction and ‘x’ represents a sample's feature(s).  A sample may have many features, each with its own particular slope (here called a weight or coefficient).   The resulting formula looks like this: 

y = w[0]*x[0] + w[1]*x[1] + … + w[p]*x[p] + b 

The equation for binary linear classification is almost identical, but returns a boolean rather than a number: 

 y = w[0]*x[0] + w[1]*x[1] + … + w[p]*x[p] + b > 0

There are different approaches for multiclass linear classification, but the One vs Rest approach involves computing a binary classification model for each individual class, then forming predictions by comparing all binary models to determine a 'winner'.  

In all cases, the model's primary task is to learn the values of ‘w’ (coefficients) and ‘b’ (intercept) which best describe the given dataset.  The predicted response ‘y’ can be thought of as a weighted sum of the sample’s features. 

Linear models are good starter algorithms.  They are fast to train and predict, scale well, work with different types of data, and are relatively easy to understand.  Caveats are that they don’t perform well on low-dimensional models, and they may be prone to overfitting on high-dimensional models.  Overfit models can be adjusted by using regularization techniques to adjust the values of the coefficients.


Naive Bayes Classifiers:  
A family of classifiers often used to model high-dimensional, sparse data such as text.  These classifiers are very similar to linear models, but they learn by looking at each feature individually and collecting per-class statistics from each feature.  Naive Bayes models have similar pros and cons to linear models, but they do not have a regression variant.


Decision Trees: 
Decision trees are recursive algorithms that construct if/else trees which sort samples into various leaf nodes.  The task of these models is to discover the questions (called 'tests') which most efficiently route samples into their requisite leaf nodes. 

The model 'learns' its tree by identifying the most prominent feature split for the given dataset (often a x < y comparison).  It then identifies the next most prominent feature split on the two subsequent regions, and so on until all regions are 'pure' leaf nodes of matching samples.

A tree composed of entirely pure leaves will often be overfitted.  To counter this, a tree may be 'pre-pruned' (recursion halted early) by declaring a max depth, a max number of leaves, or a minimum count of samples per leaf to continue splitting.

Trees are convenient because they require no data preprocessing or normalization, are invariant to data scaling, and are easy to visualize up to a certain depth.  Caveats are that they have a tendency to overfit even when pre-pruned, and they cannot make predictions beyond the range of their training data (all tree-based regressions will just keep predicting the last known point if they hit the edge of their known values).  


Ensemble Methods: 
Ensemble methods combine multiple single models to create more powerful aggregate ones.  Two examples are "Random Forests", which construct many decisions trees in order to average results across the entire 'forest', and "Gradient Boosted Regression Trees", which construct many decision trees in a serial manner, with each tree attempting to improve its predecessor.  




## Example2
## Third Example


 


