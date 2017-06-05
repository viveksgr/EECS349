**Predicting neural spike trains from two-photon calcium imaging**

Maite Azcorra-Sedano, Han Jiang, Torben Noto, Vivek Sagar

Northwestern University, EECS349 Spring 2017 - Machine Learning

Abstract

A current challenge in neuroscience is determining the spiking of neurons from recordings of features that are related to their firing, but which are naturally noisy and imprecise. Calcium imaging is a popular technique that optically measures intracellular levels of calcium from thousands of neurons simultaneously. Despite several advantages of calcium imaging, it suffers from the drawback that the calcium levels only provide a proxy for neuronal firing. Even though there is a strong biophysical framework to explain how neuronal firing relates to calcium currents, it is not clear how we can mathematically calculate the neuronal spiking from calcium signals due to factors like limited sampling rates and dye-buffering. A precise and fast algorithm for doing this would obviate the need for jointly calibrating the electrophysiological and imaging experiments.
Several computational models (deconvolution filters, Bayesian, biophysical and generalized linear models) have been proposed to predict the spike trains from calcium currents, but their estimation necessitates making several assumptions about the mechanism underlying the relationship between calcium currents and neuronal spiking. Here we implemented several supervised machine learning algorithms which do not require such assumptions, including logistic regression, support vector machine (SVM), gradient boosting, feedfoward neural networks, and recurrent neural networks to predict spike trains from calcium signals and their derivatives. RESULTS:

GRAPH:

Final Report

Introduction

Data Acquisition and Feature Selection

Simultaneous calcium and spike data were acquired from 5 sessions (with between 5-21 neurons in each session and each neuron yielding of the order of 30,000-80,000 time points worth of instances). Spike trains were binarized, and raw calcium signals were converted into features (Fig 1.). We use the calcium signal to derive additional attributes, namely, the history and future of calcium activity, first and higher order derivatives, and a moving average across the calcium time series. FIGURE:

Classification Methods

Logistic Regression:

Support Vector Machine: We constructed a linear SVM using the package LIBSVM to conduct a baseline test of how well the features could predict the labels.

Gradient Boosting: We built a gradient boosted decision tree classifier using XGBoost and tensorflow. This classifier builds many weakly accurate decision trees and combines their predictions into an ensemble that tends to be fairly accurate. We are using a binary, sigmoidal loss function and the default boosting parameters (no regularization, no maximum depth, etc.). Because of this, the model overfits the training data and performs at about a ZeroR classifier level on testing data. We plan to manipulate these values to reduce overfitting Specifically, we will focus on tuning the L1 regularization, L2 regularization, L2 regularization on tree biases, max depth, gamma, eta, and number of training iterations.

Neural Networks: We have constructed a feed forward neural network with 3 hidden layers using Tensorflow and sklearn. The network is trained for a binary classification task. The loss function being used is the probability error (cross entropy softmax with logits). We use Adam optimizer with a learning rate of 0.01 to minimize this loss. The network then predicts the spikes given the test examples. 

Results

