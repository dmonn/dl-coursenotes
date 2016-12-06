# Neural Networks

Charles and Michael from Georgia Tech

## Introduction

A neural network consists of multiple neurons. Each neuron takes some input `X1, X2, X3, ..` multiplied by a weight `w1, w2, w3, ..` and returns an output `y`.

A **Perceptron** for example. Takes some input, multiplies it with the weight, then sums all results up and checks, wether or not the result is above a "firing threshold." If it is, it returns 1, if not, it returns 0. It was first developed in the 50s!

## Perceptron

The perceptron unit gets some input and then computes the **activation**. The threshold of the activation is the output.
With this perceptron unit, we can use the boolean function AND, OR and NOT. However, if we want to form something like XOR, we need more than one unit. We need another layer! So we have a **Perceptron Network**

### Training

In machine learning, you want an alghorithm to find the weights depending on your training data. There are two ways to do this: The perception rule and the gradient descent.

With the perception rule, we start off with some weights `w` and we are going to iterate and change these weights over and over again. We are changing the weight by a delta weight defined by the following alghorithm:

`Δw = η(y - y')x`

With y as the target, y' as the output, η as the learning rate and x as the input.
This only works, when the data is "Linearly Seperatable". This means that you could draw a line on a chart to seperate 0 and 1 values.

The Gradient descent is more robust when it comes to non-linear data.

#### Comparison

Perceptron: `Δw = η(y - y')x`

Gradient Descent: `Δw = η(y - a)x`

We can see that the only difference is, that we subtract the activation value in the gradient descent, while we use the thresholded value with he perceptron rule.
The perceptron rule has a guarantee of a finite convergence when we have linear seperability. The gradienct descent is more robust to data sets that are not linear seperatable but it is only going to converge to a local optimum.

## Sigmoid Function

`sigma(a) = 1 / (1 + e^(-a))`

As the activation gets less, the sigmoid is going towards zero. As the activation gets bigger, the sigmoid goes towards one.

## Optimizing weights

* Momentum: To "continue in the direction we were going". Basically bouncing out of local minimas
* Randomized Optimization
* Penalty for Complexity, less layers and nodes, smaller numbers!

## Terms

* Restriction Bias: Set of Hypothesis we are considering, e.g. only sigmoid functions etc. What should it do?
* Preference Bias: Alghorithm you are using. Selection of one representation over another. What Alghorithm do we use? Also, the initialisation of the weights. -> often small random values to avoid local minimas

## Summary

* Perceptrons are threshold units
* Networks can produce any boolean function
* Perception rule - Finite time for linearly seperable
* General differentiable rule - Basic Backpropagation & Gradient descent
* Preference / Restriction Bias
