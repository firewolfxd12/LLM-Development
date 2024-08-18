# Intro to neural networks and backpropagation: building micrograd

Created by: Nick Durbin
Created time: July 22, 2024 7:06 PM
Last edited by: Nick Durbin
Last edited time: July 25, 2024 1:59 AM
Tags: Micrograd

# Micrograd overview

- Micrograd is an autograd engine that implements backprapogation
- Backpropagation - an algorithm that efficiently **evaluates the gradient** of a loss function with respect to the weights of a neural network.
    - It allows to iteratively tune the weights of the neural network to minimize the loss function and therefore improve the accuracy of the network
    - It is at the mathematical core of any modern deep neural network library like PyTorch or Jaxx
- Micrograd allows to build mathematical expressions using Value Objects (a class in Python).
    - Under the hood it builds an expression graph to then apply the **chain rule from calculus**
    - a + b = g | a.grad will allow to understand how g will be affected if a is tweaked a tiny amount in the positive direction
- Neural Networks are a certain class of mathematical expressions
    - They take the input data and the weights of a NN as an input
    - And the output are the predictions of the neural net or the **loss function**
- Backpropagation is more general than NN’s it can be used on any mathematical expressions. They just happen to be used to train NN’s
- Micrograd works on scalar values and breaks down NN’s for educational reasons
- Micrograd is what you need to train NN’s and everything else is just efficiency

# Derivative of a simple function with one input

- The derivative of a function of a single variable at a chosen input value, when it exists, is **the slope of the tangent line to the graph of the function at that point**.

Didn’t take notes for the rest of the video. Here was the Jupiter Notebook during the lecture:

[nn-zero-to-hero/lectures/micrograd at master · karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)