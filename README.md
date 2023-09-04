# Basic of Machine Learning: Exploring Strategies for Learning from Data

A series of projects exploring popular machine learning strategies.

## 1. k-Nearest neighbors with RMSE loss, least squares with singular value decomposition (SVD) (knn_svd.ipynb)

### k-Nearest Neighbors (k-NN)

The **k-NN algorithm** attempts to generate a set of predictions for a dataset by taking the average values of a given test data point's "nearest neighbors," the data points in the training set which are "closest" to the test point as determined by a specified distance metric. The number of "neighbors" considered is given by the variable **k**. As mentioned above, this approach will be assessed with an **RMSE loss function**.

### Least Squares Method

The **least squares approach** attempts to find a set of weights (w) that best describe the relationship between the training x (plus a bias term) and the y training values. This is done by attempting to minimize the least-squares function.

Singular value decomposition, or **SVD**, is a method for solving overdetermined systems of equations by decomposing the X matrix into three unique matrices: two orthogonal (U,V), and one diagonal (Σ). In this exploration, the economy-SVD (that with extra rows or columns of zeros removed) will be used in the least-squares formula.

## 2. General linear models, particularly as they apply to regression and classification problems (glm.ipynb)

The purpose of this report is to explore concepts and techniques related to **general linear models**, particularly as they apply to regression and classification problems. The objectives of this report are as follows:

### Manipulating General Linear Models

The report aims to provide familiarity with the process of manipulating general linear models to determine their weights, primarily by minimizing an associated loss function. In this process, the following key concepts will be explored:

#### Tikhonov Regularization

This technique is used for regularizing ill-posed problems.

#### Dual Representation

The technique of rearranging and representing a general linear model in such a way that allows for the implementation of the 'kernel trick,' reducing the computational complexity of weight-solving.

### Implementation of RBF Regression

The report will include the manual implementation of a simple algorithm for radial basis function (RBF) regression. Radial basis functions, specifically Gaussian RBFs, will be explored. The effect of shape and regularization parameters on the generated loss will be assessed.

### Greedy Algorithm for Model Basis Functions

The report will discuss the manual implementation of a greedy algorithm for selecting the model's basis functions to improve model sparsity. Achieving model sparsity is desirable for various reasons, including reductions in computational, memory, and storage requirements.

