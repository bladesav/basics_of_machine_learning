# Basic of Machine Learning: Exploring Strategies for Learning from Data

A series of projects exploring popular machine learning strategies.

## 1. k-Nearest neighbors with RMSE loss, least squares with singular value decomposition (SVD) (knn_svd.ipynb)

### k-Nearest Neighbors (k-NN)

The **k-NN algorithm** attempts to generate a set of predictions for a dataset by taking the average values of a given test data point's "nearest neighbors," the data points in the training set which are "closest" to the test point as determined by a specified distance metric. The number of "neighbors" considered is given by the variable **k**. As mentioned above, this approach will be assessed with an **RMSE loss function**.

### Least Squares Method

The **least squares approach** attempts to find a set of weights (w) that best describe the relationship between the training x (plus a bias term) and the y training values. This is done by attempting to minimize the least-squares function.

Singular value decomposition, or **SVD**, is a method for solving overdetermined systems of equations by decomposing the X matrix into three unique matrices: two orthogonal (U,V), and one diagonal (Σ). In this exploration, the economy-SVD (that with extra rows or columns of zeros removed) will be used in the least-squares formula.

## 2. General linear models, particularly as they apply to regression and classification problems (glm.ipynb)

The purpose of this project is to explore concepts and techniques related to **general linear models**, particularly as they apply to regression and classification problems.

### Manipulating General Linear Models

This project aims to provide familiarity with the process of manipulating general linear models to determine their weights, primarily by minimizing an associated loss function. In this process, the following key concepts will be explored:

#### Tikhonov Regularization

This technique is used for regularizing ill-posed problems.

#### Dual Representation

The technique of rearranging and representing a general linear model in such a way that allows for the implementation of the 'kernel trick,' reducing the computational complexity of weight-solving.

### Implementation of RBF Regression

Radial basis functions, specifically Gaussian RBFs, will be explored. The effect of shape and regularization parameters on the generated loss will be assessed.

### Greedy Algorithm for Model Basis Functions

Manual implementation of a greedy algorithm for selecting the model's basis functions to improve model sparsity. Achieving model sparsity is desirable for various reasons, including reductions in computational, memory, and storage requirements.

## 3. Statistical data-learning approaches (map_dnn.ipynb)

The purpose of this project is to explore concepts and techniques related to **Maximum A Posteriori Estimate (MAP)** and **Deep Neural Networks**. The project objectives are as follows:

### Maximum A Posteriori Estimate (MAP)

In this project, a logistic regression approach is used to learn the weights of a modified binary version of the iris dataset. The project evaluates the use of both full-batch gradient descent and stochastic gradient descent to find the MAP estimate of the parameters.

### Deep Neural Networks

The project assesses the capacity of a fully-connected neural network with two hidden layers and 100 neurons. This network is trained via stochastic gradient descent with a minibatch of 250. The evaluation focuses on the network's ability to predict on the MNIST image dataset.

## 4. Bayesian techniques for regression and classification (Bayes.ipynb)

The purpose of this project is to explore and assess the use of Bayesian techniques for classification and regression tasks. The project objectives are as follows:

## Bayesian Model Assessment

# Project Summary

The purpose of this project is to explore and assess the use of Bayesian techniques for classification and regression tasks. The project objectives are as follows:

## Bayesian Model Assessment

In this project, we aim to assess the effect of prior variance on models through the calculation of log marginal likelihoods. We will use Laplace estimation as a method to reduce the computational complexity of the process.

### Importance Sampling for Posterior Predictions

The project involves employing importance sampling in posterior class predictions by selecting an appropriate proposal distribution.

### Bayesian Linear Model for Time-Series Data

We will assess the efficacy of a Bayesian linear model in predicting time-series data. Additionally, we will evaluate methods to improve the performance of this approach.
