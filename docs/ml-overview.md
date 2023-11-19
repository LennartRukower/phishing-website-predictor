# Machine Learning Techniques
## K-nearest Neighbour
K-NN classifies a data point based on how its neighbors are classified. It looks at the 'K' nearest neighbors (where 'K' is a specified number) and classifies the new point based on the majority class among these neighbors.
**Advantages**:
    - Simple and easy to implement.
    - No assumptions about data — useful for non-linear data.
    - Model adapts immediately as we collect new training data.
**Disadvantages**:
    - Computationally expensive — especially as data grows, because it searches for nearest neighbors.
    - High memory requirement.
    - Sensitive to irrelevant features and the scale of the data.

## Logistic Regression
Logistic Regression uses a logistic function to model a binary dependent variable (extensions exist for multiple categories).
**Advantages**:
    - Efficient and easy to implement.
    - Provides probabilities for outcomes.
    - Works well for linearly separable data.
**Disadvantages**:
    - Assumes linearity between dependent and independent variables.
    - Not as powerful for complex relationships in data.
    - Can be vulnerable to overfitting.

Further Resources:
https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

## Naive Bayes Classification
This technique is based on applying Bayes' theorem with the assumption of independence between every pair of features. Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
**Advantages**:
    - Works well with high-dimensional data.
    - Efficient and easy to implement.
    - Performs well with categorical input variables compared to numerical variables.
**Disadvantages**:
    - Based on the assumption that features are independent, which is rarely the case.
    - Not the best choice for regression.

## Decision Trees
A decision tree is a tree-like model of decisions. It consists of nodes, branches, and leaves. To classify a new object or predict a value, you start at the root of the tree and take the decisions prescribed by the internal nodes until a leaf node is reached, which gives the prediction.
**Advantages**
- Easy to Understand and Interpret
- Requires Little Data Preparation
- Handles Both Numerical and Categorical Data
- Non-Parametric Method
**Disadvantages**
- Decision trees are prone to overfit the training data, making them less generalized. This can be mitigated by techniques like pruning, setting a minimum number of samples per leaf, or limiting the depth of the tree.
- Small variations in the data might result in a completely different tree. This can be mitigated by using decision trees within an ensemble method, like Random Forests.
- If some classes dominate, decision trees can create biased trees. It’s therefore recommended to balance the dataset before creating the tree.
- Poor Performance on Continuous Variables

## Support Vector Machines
SVM is a powerful classifier that works by finding a hyperplane that best separates the classes in the feature space. If the data is not linearly separable, it uses a kernel trick to transform the data into a higher dimension where a hyperplane can be used.
**Advantages**:
    - Effective in high-dimensional spaces.
    - Memory efficient as it uses a subset of training points.
    - Versatile: different kernel functions can be specified for the decision function.
**Disadvantages**:
    - Not suitable for large data sets because of its high training time.
    - Less effective on noisier datasets with overlapping classes.
    - Requires careful tuning of parameters and selection of an appropriate kernel.

## Neural Networks
Neural networks consist of layers of interconnected nodes, each node being a simple processor. The data is fed into the input layer, and then processed through one or more hidden layers using weights that are adjusted during learning. The output layer produces the prediction.
**Advantages**:
    - Highly flexible and can model complex non-linear relationships.
    - Can learn features and tasks directly from data.
    - Works well with large datasets.
**Disadvantages**:
    - Requires a large amount of data to perform well.
    - Computationally intensive.
    - Model interpretation can be difficult (often referred to as a "black box").


# Evaluation of ML techniques
## K-nearest Neighbour
- Simple Technique
- But: Curse of dimensionality, Sensitive to irrelevant features, computationally expensive 

## Logistic Regression
- Efficient for binary classification tasks
- But: Might struggle with complex, non-linear relationships, Requires feature scaling
## Naive Bayes
- Effective on large feature spaces
- But: Assumes feature independence, Performance can degrate 

## Decision Trees
- Can handle mix of discrete, continuous, binary, and categorical features
- Interpretablle
- But: Overfitting Risk, imbalanced data can increase bias 

## Support Vector Machines
- Effective in high-dimensional spaces
- Good for binary clasification
- But: Need for careful tuning of parameters, can be computational intensive

## Neural Networks
- Highly flexible
- Can automatically detect significant features
- But: Complex to setup and train, Need "a lot of data", "Black box"

## Result
Use Logistic Regression, Decision Trees (start with simple tree and then use ensemble methods, regularization and pruning - when data is imbalanced use SMOTE), SVM and Neural Networks for binary classification of websites.