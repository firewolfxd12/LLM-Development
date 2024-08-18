# Building makemore Part 5: Building a WaveNet

Created by: Nick Durbin
Created time: July 28, 2024 3:31 PM
Last edited by: Nick Durbin
Last edited time: July 28, 2024 7:02 PM
Tags: Makemore

The starter code is very similar to the one in Part 3.

## Lets fix the learning rate plot

Currently our learning rate plot looks like this

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled.png)

As we can see it is very thick and should be changed. We can make it better by plotting the averages on each 200 iterations. 

```python
torch.tensor(lossi).view(-1, 1000).mean(1)
```

We have 200000 iterations during training in total. The `view` function divides the `lossi` accumulated during training into a [200, 1000] array. Then the `mean` function gets the average per row. When we plot it we get the following result.

```python
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
```

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%201.png)

## PyTorch-ifying the code

The forward pass can be simplified to have less lines of code

```python
# forward pass
emb = C[Xb] # embed the characters into vectors
x = emb.view(emb.shape[0], -1) # concatenate the vectors
for layer in layers:
	x = layer[x]
loss = F.cross_entropy(x, Yb)
```

To do this we can add two more modules for the embedding operation and the concatenating operation.

```python
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]
```

```python
class FlattenConsecutive:
    
  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out
  
  def parameters(self):
    return []
```

Both of these classes exist in PyTorch. Now these layers can be included in the list

```python
layers = [
	Embedding(vocab_size, n_embd),
	Flatten()
	...
]
```

The forward pass now looks like this

```python
# forward pass
x = Xb
for layer in layers:
	x = layer(x)
loss = F.cross_entropy(x, Yb)
```

We can further simplify the code by making our own `Sequential` module similar to the one in the PyTorch library.

```python
class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]
```

Now instead of having a list `layers` we can make a sequential called `model`.

```python
model = Sequential([
	Embedding(vocab_size, n_embd),
	Flatten()
	... more layers here...
)]
```

The forward pass now looks like this

```python
# forward pass
logits = model(Xb)
loss = F.cross_entropy(logits, Yb)
```

Thanks to the new modules, we can make the code overall cleaner and more readable.

## Overview: WaveNet

Currently we only have one hidden layer and it doesn’t make sense to squash so much information into one layer. In WaveNet the information is fused slowly throughout the layers as it gets deeper. 

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%202.png)

The context is processed as bigrams in the beginning and then on the third layer 2 bigrams get fused also until they reach the output. 

So the goal is to make the network deeper and on each level we will only fuse two consecutive elements.

## Changing the context length to 8 instead of 3

After changing the context length, the loss function improves, although squashing so much info into a single hidden layer isn’t that good. The model also has much more parameters and the training sets look different.

## Implementing WaveNet

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%203.png)

- If we sample 4 examples out of the training set we see that the context length is now 8 and the array size is [4, 8].

### Shapes of the layers:

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%204.png)

The Embedding layer converts the examples into 10-dimensional embeddings so it can be fed to the network causing the shape to be [4, 8, 10]. If a [1, 8] example is used to index into this array, its embedding will be returned.

The Flatten layer basically stretches the examples into a long row so that the matrix multiplication works. It works similarly to the concatenation operation causing the array to be [4, 80], but the `view` function is used since it is computationally cheap and achieves the same result.

The Linear layer takes 80 and converts it into 200 channels via matrix multiplication. The shape [4, 200] results from what happens inside the layer:

```python
(torch.randn(4, 80) @ torch.randn(80, 200) + torch.randn(200)).shape
```

Matrix multiplication in PyTorch is quite powerful. If we add more dimensions to the first tensor, the @ operation will still work. 

```python
(torch.randn(4, 5, 2, 7, 80) @ torch.randn(80, 200) + torch.randn(200)).shape
```

The shape of this operation will be [4, 5, 2, 7, 200], where the first 4 dimensions are completely unchanged.

Because of how WaveNet operates we now want to feed two characters from the 8 at a time to then fuse them together. 

```python
(torch.randn(4, 4, 20) @ torch.randn(20, 200) + torch.randn(200)).shape
```

In this layer we are now feeding the same amount of examples but now in pairs, since there are 10 numbers for each character in the embedding. `torch.randn(4, 4, 20)` this will divide the examples into our pairs and `torch.randn(20, 200)` will fuse them in the first layer.

Now we have to update our `Flatten` module to support this kind of separation.

```python
e = torch.randn(4, 8, 10) # goal: we want this to be (4, 4, 20) where consecutive 10-d vectors get concatenated
explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
explicit.shape = [4, 4, 20]
```

This can be done the following way:

```python
class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []
```

`squeeze` removes any dimensions that are equal to the specified one. `//` is integer division.

Using this module we can now make a 3 layer neural net with way more parameters.

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%205.png)

Here are the dimensions of the layers:

![Untitled](Building%20makemore%20Part%205%20Building%20a%20WaveNet/Untitled%206.png)

We end up with outputting [4, 27] with the probability distributions for the next character.

We have basically implemented WaveNet only with a block length of 8 not 16.

## Fixing the batchnorm1d bug

In the current implementation, the batchnorm1d module assumes that it gets a 2 dimensional tensor. With our new model batchnorm1d will receive a 3 dimensional tensor also. This means that the `mean` and `variance` isn’t being calculated correctly, only using a subset of the batch, not the whole batch. This can be fixed by adding an if else statement that assigns the dimensions correctly.

```python
if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
```

## Re-training WaveNet with no bugs

After re-training we gained a slightly better improvement, but not that significant. If we scale up the model, the performance will be much better, but training will take more time.

We did not implement the WaveNet model completely, since there is a more complicated linear layer that we didn’t change. Some of them are residual connections, skip connections and etc.

## WaveNet but with dilated casual convolutions

The use of convolutions in the paper is strictly for efficiency, it didn’t actually change the model that we have already. 

A convolution is a “for loop”. It allows us to forward Linear layers efficiently over a space. It allows to calculate all of the outputs as an input “slides” at the same time. We will cover it more deeply in a future video.

## Torch.nn

Basically what we’ve done is reimplemented a part of the `torch.nn` library. In the future we will just start using `torch.nn` directly.

## The developmental process of deep neural networks

1. We spend a lot of time reading the PyTorch documentation
    1. We look at shapes of the inputs
    2. Types of layers
    3. What inputs can there be and what does the layer do

<aside>
💡 PyTorch documentation is unclear and incomplete.

</aside>

1. There is a lot of works with the shapes of tensors and efficiently doing operations on them
    1. Is the sequence of arrays in accordance with the documentation?
2. Spend a lot of time prototyping making sure the shapes work out in a jypiter notebook. Once the code works, implement in repository
    1. Jypiter notebook for prototyping and making sure code works as intended
    2. VSCode for initiating repository tests and upload

## Going forward

1. We will convert the neural network to use dilated casual convolutional layers, implementing the ConNet
2. Get into residual connections and skip connections and why they are useful
3. We don’t have an experimental harness, this is not representative of deep neural network workflows.
    1. You have to make an evaluation harness
    2. Kick off experiments
    3. You have arguments the script can take 
    4. Kick off experimentation
    5. Looking at plots of training and validation loss
    6. Doing hyper-parameter searches
4. Cover RNN, GRU, and Transformers too