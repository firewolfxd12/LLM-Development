# Building makemore Part 3: Activations & Gradients, BatchNorm

Created by: Nick Durbin
Created time: July 26, 2024 5:00 PM
Last edited by: Nick Durbin
Last edited time: July 27, 2024 1:24 AM
Tags: Makemore

## Introduction

In the last lecture we implemented an MLP from the Bengio paper only on character level.

We want to move on RNN, GRU and more complex models.

Before we do this we should work on the MLP a little longer. We should have a good intuitive understanding of the activations and backpropagation. This is important to understand the history of development of these models. RNN are very expressive and implement all the algorithms, but they are not as optimizable with the gradient based techniques. To understand this we have to understand the activations, gradients and how they behave during training. A lot of variants since neural networks have tried to improve optimization. Lets get started.

## Starter code

The code is like before but more cleaned up. See it here.

[https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)

There is a decorator in PyTorch that makes sure not to build the DAG that is used during backpropagation for functions. No gradient tracking

```python
@torch.no_grad()
```

## Fixing initial high loss

The loss during training on the first iteration is messed up since it is too high. When training neural nets it is important to know what the loss should be during intialization for different loss functions. 

We can calculate the initialization amount. During initialization the probability of any character should be 1/27. 

```python
-torch.tesor(1/27).log()
```

This equals about 3.29 not 27. 

On initialization the logits should be about equal to each other so that the loss is lower. If this is not made sure of, the loss might even be `inf` . But in the beginning the best number is zero and be want to record the loss at initialization.

Since we want the weights to be roughly zero, we don’t really want to add a high bias of random numbers to the weights at initialization. We can fix this by multiplying `b2` by 0 and multiplying `W2` by 0.01.

```python
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0
```

Setting all the weights to just zero might have negative effects also. Things can go very wrong. They should have some entropy.

Before the graph looked like a hockey stick and the first few iterations of the optimization were wasted on shrinking the high loss down. Since we set the weights initially to be about 0 all the iterations are now being used to get the “hard gains”.

## Fixing the saturated tanh

If you take a look at the hidden layer a lot of the values are initialized at -1 or 1. If you remember, tanh is a squashing function to get the values from [-1, 1].

Lets look at a histogram of `h` to understand the distribution of the hidden layer.

To get all the values of `h` into a list we can do this

```python
h.view(-1).tolist()
```

- -1 will make all the values be in a sone dimensional tensor.

To build a histogram we can

```python
plt.hist(h.view(-1).tolist(), 50);
```

- The `;` removes other not relevant data

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled.png)

We can see that most of the data is -1 or 1. This is because before the data was fed into `tanh` there were many values more than 1 and -1 which resulted in squashing them to the ends. This is a big problem.

If the numbers are by -1 or 1, or in the flat regions of the `tanh` , changing any values before it will not have any effect on the loss function resulting in the gradient being low. The gradients become muted at this layer.

If some neuron is in the tail of the `tanh` function, the neuron effectively becomes dead suspending its ability to learn. A neuron is dead of all of its `tanh` values are more than 0.99.

`tanh` is not the only activation function. 

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled%201.png)

Dead neurons may just happen by chance either at initialization or during training. You can get a dead `ReLu` neuron also. If a neuron never activates for any example given in the dataset, then that `ReLu` neuron will never learn. 

This issue usually comes if there are flat parts in the activation function. 

The hidden layer initialization is too saturated and should be fixed. This can be done the following way.

```python
  hpreact = embcat @ W1 + b1
...
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0
```

- `hpreact` gets multiplied by `W1` .
- We can change `W1` at initialization

After this change we went from half white half black to mostly black

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled%202.png)

With the following distribution

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled%203.png)

After the fixes the results are way better

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled%204.png)

Even though the initialization was terrible, the network still eventually learned. This is not the case usually. As the network grows these problems stack up. The deeper the network with more layers, the less forgiving some of these errors are.

## Calculating the init scale

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200)
y = x @ w
```

![Untitled](Building%20makemore%20Part%203%20Activations%20&%20Gradients/Untitled%205.png)

Because of `randn` the mean and standard deviation are about 0. After multiplication the standard deviation becomes 3. The goal is too figure out how to make it to remain a Gaussian after multiplication. 

The way to do this is to divide by the square root of the fan in.

```python
w = torch.randn(10, 200) / 10 ** 0.5
```

After multiplication the standard deviation becomes one. For the `ReLu` function since it throws away half of the inputs that are less than zero, you have to multiply by (2 / amount of inputs) ** 0.5 to preserve the Gaussian. This is called the gain.

There is a function in PyTorch that initializes neural networks called `nn.init.kaiming_noraml_` . It takes in a nonlinearity so that the gain gets changed accordingly. 

The gain is important because nonlinearities are contractive functions. In order to fight the contracting we have to boost the weights to preserve the nonlinearity.

The PyTorch library added this function because of a paper that studied this. 

Modern innovations make neural network initializations less important, meaning it doesn’t have to be exactly right to work. Some of these innovations are:

- residual connections,
- the use of nomalization layers
    - batch normalization
    - layer normalization
    - group normalization
- optimizers that are more complex than gradient descent
    - RMS prop
    - Adam

These modern optimizations make it less important to precisely initialize a neural net.

Since our weights are initialized by `randn` and we multiplied by 0.2, our standard deviation is 0.2. Andrej usually just divides by the square root of fan in. If we want to do it like it is done in PyTorch we have to multiply the square root by the gain of the appropriate nonlinearity.

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)((n_embd * block_size) ** 0.5)
```

That is the kaiming init. We got the value by calculating it and not just guessing.

## Batch Normalization

This idea came from a paper from Google and it made it possible to train neural nets quite reliably. The reason why we want to make the batches gaussian is to reduce the chance of the nonlinearity killing neurons.  

We want hpreact to be roughly gaussian. This can be done by just normalizing these states. This operation is differentiable. We can just standardize these activations so that they are exactly gaussian. 

Normalizing the batch to unit gaussian can be done like this

```python
hpreact = (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True)
```

All these operations are perfectly differentiable and you can just train this.

You wont achieve a good result with this. We want them to be gaussian but only at initialization. We want to allow the neural net to make it more diffuse or sharp and not force it to be gaussian. The distribution should move around and be controlled by the backpropagation. 

This can be done if we should add a batch normalization gain and bias. 

```python
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
```

Now we can multiply and offset the normalization

```python
hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
```

Since we have the gain set to one and bias to zero, at initialization each neurons firing values in this batch will be exactly unit gaussian, no matter what the numbers are in `hpreact`. 

Thanks to the gain and bias, the network will be able to backpropagate and move the way it wants internally. 

Since the network is small the batch normalization is no doing much here. If you get a much deeper neural net it will become very difficult to tune the weight matrices activation throughout the neural net are roughly gaussian. 

It is costmary to take these linear layer or convolutional layer and append a batch normalization layer right after. These layers might be placed throughout the net and it significantly stabilizes the training which is why there layers are popular.

The stability offered comes at a terrible cost. Because of the normalization through the batch, we are coupling these batch examples mathematically in the forward pass and backward pass of the neural net. The hidden state activations and the logits for any input example are not just a function of that example and its input, but also a function of all the other examples that happen to come for a ride in the batch, and they are sampled randomly. The activations will change slightly depending what other examples are in a batch. The hidden layer is going to jitter because of the changes in the standard deviation, which will cause the logits to jitter. This turns out to be good in neural network training as a side effect. 

By introducing this noise, the network will find it hard to overfit to the data. 

Since batch data gets stuck in the network, people have been trying to move to other forms of regularization that do not couple batches. These are:

- layer normalization
- instance normalization
- group normalization etc.

Well cover some of these later. Batch normalization was one of the first to be introduced, it worked extremely well:

- It has a regularizing effect
- stabilized training

People tried moving to other normalization tequtiques, but its been hard because it worked well and had a good effect on the distributions of activations. 

Here are one of the strange outcomes of batch normalization. We would like to feed the neural net a single example and then get a prediction. 

How do we do this if in the forward pass the net estimates the statistics of the mean and standard deviation of a batch? The neural net expects batches as an input now. 

The solution is to a have a step after training that calculates and sets the batch norm mean and standard deviation a single time over the training set.  

```python
# calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 # + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnstd = hpreact.std(0, keepdim=True)
```

We can calculate the bnmean and bnstd a single time and then use those values during inference

But it is better to calculate these values in a running manner not as a separate step. We can get the average of`bnmean` and `bnbias` and remember it.

```python
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
```

We should update these values without gradients inside the training loop.

```python
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```

These values are very similar to the ones that we calculated after training so now we can calculate it without a second step.

Since we have a bias in the bn layer we can remove the `b1` biases since it will get subtracted out anyway.

```python
hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation
```

The batch normalization layer is effective at controlling the statistics of deep neural nets.

## PyTorch-ifying the code

`torch.nn` has many layers that you can implement. These include the `linear` , `batchnorm1d`, layers also. 

```python
# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd),            generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ]

with torch.no_grad():
  # last layer: make less confident
  layers[-1].gamma *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 #5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Notes:

`Tanh` nonlinearities allow us to turn a neural net from a linear function into a neural network that can approximate any function.

## Graph analysis

[01:18:35](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=4715s)

just kidding: part2: PyTorch-ifying the code

[01:26:51](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=5211s)

viz #1: forward pass activations statistics

[01:30:54](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=5454s)

viz #2: backward pass gradient statistics

[01:32:07](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=5527s)

the fully linear case of no non-linearities

[01:36:15](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=5775s)

viz #3: parameter activation and gradient statistics

[01:39:55](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=5995s)

viz #4: update:data ratio over time

[01:46:04](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4&t=6364s)

bringing back batchnorm, looking at the visualizations

## Summary

1. Introduction to batch normalization - one of the first modern normalizations that helped stabilize neural networks
2. PyTorch-ifying code
3. Using graphs to analyze health of neural network