# Building makemore Part 2: MLP

Created by: Nick Durbin
Created time: July 24, 2024 4:57 PM
Last edited by: Nick Durbin
Last edited time: July 26, 2024 4:19 PM
Tags: Makemore

## Introduction

Hello everyone. Today we are continuing our implementation of makemore. In the last lecture we implemented the bigram language model. We implemented it using counts and using a super simple neural network that has a single linear layer. 

The way we approached this is that we looked at the single previous character and we predicted the distribution that would go next in the sequence. We did that by taking counts and normalizing them into probabilities so that each row summed to one. This is all well and good if you only have one character of previous context. It works and it is approachable. The problem with this model is that the predictions of the model are not very good because it can only take one character of context. The model didn’t produce very name-like sounding things. 

Now the problem with this approach though if we would take more context into account than predicting the next character in a sequence things quickly blow up and the size of the table grows exponentially with the length of the context. If we only take a single character at a time that is 27 possibilities of context, but if we take two characters and try to predict the third one suddenly the number of rows in the matrix is 27*27 = 729 possibilities. If we take 3 characters as the context suddenly we have 19 683 possibilities. The whole thing explodes and doesn’t work very well.

## Bengio et al. 2003 (MLP language model) paper walkthrough

Today we are going to implement a Multilayer perceptron model to predict the next character in a sequence. We are building a character level language model, in this paper they have a vocabulary of 17 000 possible words and they instead build a word level language model. We are still going to stick with the characters but take the same modeling approach. 

What they do is associate to every word a 30 dimensional feature vector. Every vector is embedded into a 30 dimensional space. They have 17 000 points or vectors in a 30 dimensional space and you might imagine that it is very crowded that is a lot of points for a small space. In the beginning the words are spread out at random, but then we are going to tune these embeddings using backpropagation. So during the course of training these points are going to move around in this space and you might imagine that words that have very similar meanings or that are indeed synonyms of each other might end up in a very similar part in the space and conversely very different things would go somewhere else in the space. 

Now their modeling approach otherwise is identical to ours. They are using a multi layer neural network to predict the next word given the previous words and to train the neural network they are maximizing the log likelihood of the training data just like we did. So the modeling approach itself is identical. 

Now they have a concrete example of this intuition. Why does it work? Basically suppose that for example you are trying to predict a “dog was running in a ___”. Now suppose that the exact phrase “a dog was running in a ___” has never occurred in a training data and here you are at sort of test time layer when the model is deployed somewhere and it’s trying to make a sentence and it’s saying “a dog was running in a ___” and because it’s never encountered this exact phrase in the training set you’re out of distribution as we say. You don’t have fundamentally any reason to suspect what might come next but this approach actually allows to get around that because maybe you didn’t see the exact phrase but maybe you’ve seen similar phrases. Maybe your network has learned that “a” and “the” are frequently interchangeable with each other and so maybe it took the embedding for “a” and the embedding for “the” and put them nearby each other in the space. So you can transfer knowledge through the embedding and generalize in that way. Similarly the network could know that cats and dogs are animals and co-occur in lots of very similar contexts. Even though you haven’t seen the exact phrase you can through the embedding space transfer knowledge and generalize to novel scenarios. 

![Untitled](Building%20makemore%20Part%202%20MLP/Untitled.png)

Now lets see the diagram of the neural network. In this example we are taking three previous words and we are trying to predict the fourth word in a sequence. These three previous words are out of a vocabulary of 17 000 possible words. Every word is fed using an index of the word, and because there are 17 000 words the index is an integer between 0 and 16 999. There is also a lookup table called C. This lookup table is a matrix that is 17 000 by let say 30. We are treating this as a lookup table and for every index is plucking out a row of this embedding matrix so that each index is converted to the 30 dimensional vector that corresponds to the embedding vector for that word. 

Here we have an input layer of 30 neurons for three words making up 90 neurons in total. This matrix C is shared across all the words. We are always indexing into the same matrix C over and over for each words. 

Next up is the hidden layer of this NN. The size of the layer is a hyper parameter. This means that the parameter is up to the designer of the NN. It can be as large or small as you like. We are going over multiple choices of the size of the hidden layer and we are going to evaluate how well they work. 

So say there are 100 neurons in the layer. All of them would be fully connected to the 90 numbers that make up the 3 words. This is a fully connected layer and there is a tanh non linearity and an output layer. Because there are 17 000 possible words that could come next, the output layer has 17 000 neurons that are fully connected to the neurons in the hidden layer. There are a lot of parameters because there are a lot of words. Most computation happens here and it is the most expensive. 

There are 17 000 logits in the output layer and on top of it we have the soft max layer which we seen before. Every one of these logits is exponentiated and everything is normalized to sum up to one so that we have a nice probability distribution for the next word in the sequence. During training we have the label, the identity of the next word in the sequence. That word or its index is used to pluck out the probability of that word and then we are maximizing the probability of that word with respect to the parameters of the neural net. The parameters are the weights and biases of the output layer, the weights and biases of the hidden layer, and the embedding lookup table C. And all of this is optimized using backpropagation. These dashed arrows ignore those. That represents a variation of the neural net that we wont explore. This is the setup and now lets implement it. 

## (re-)building our training dataset

For this we are importing PyTorch:

```python
import torch
import torch.nn.functional as F
```

We have to read all the names into a list of words like before

```python
# read all the words
words = open("names.txt", "r").read().splitlines()
```

Then we have to build out the vocabulary of characters from all the characters as strings to integers and vice versa

```python
# build the vocabulary of characters and mappings, to/from integers
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
```

The first thing we should do is compile the data set for the neural network. We had to rewrite the code.

```python
# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []

for w in words:
	
	context = [0] * block_size
	for ch in w + ".":
		ix = stoi[ch]
		X.append(context)
		Y.append(ix)
		context = context[1:] + [ix] # crop and append
		
	X = torch.tensor(X)
	Y = torch.tensor(Y)
```

`X` are the input to the neural net

`Y` are the labels for each example inside `X`

The `block_size` can be changed to whatever value is needed

`X` will have `block_size` amount of integers that we converted from characters using the dataset,  and `Y` will have a single integer corresponding to what is the prediction for the integers in `X` .

## Implementing the embedding lookup table

We have 27 possible characters and we are going to embed them in a lower dimensional space. In the paper they had 17 000 words and they embed them in spaces as small as 30 dimensions. In our case we only have 27 possible characters so lets cram them in a 2D space.

```python
C = torch.randn((27, 2))
```

Before we embed all of the integers inside the input `X` using the lookup table `C` . Let us try to embed a single individual integer lets say 5.

One way this works is we can just take the `C` and index into row 5

```python
C[5]
```

This gives us the 5th row of `C` . This is one way to do it. The other way that we presented in the previous lecture is actually seemingly different but identical. What we did is we took the integers an used the one-hot encoding.

```python
F.one_hot(5, num_classes=27)
```

This is a 26 dimensional vector with all zeros except the 5th bit is actually turned on. This actually doesn’t work the reason is that 5 should be a tensor.

```python
F.one_hot(torch.tensor(5), num_classes=27)
```

Now notice that that if we take this one-hot vector and multiply it by `C` . 

```python
F.one_hot(torch.tensor(5), num_classes=27) @ C
```

First you would expect an error because the one-hot returns a tensor with a int64 data type. `C`  is a float32 tensor. PyTorch doesn’t know how to multiply and int with a float. So we have to cast it to a float so that we can multiply

```python
F.one_hot(torch.tensor(5), num_classes=27).float @ C
```

The output is identical to C[5] because of how matrix multiplication works. We have a one-hot vector multiplied by the columns of `C` . Because of the zeros we end up masking out everything except the 5th row which is plucked out arriving to the same result. 

In the diagram before we can think of it as an integer indexing into a lookup table `C`  but you can also think of it as a first layer of a bigger neural net. This layer has neurons that have no non linearity, there is no tanh, and their weight matrix is `C`  and we are encoding integers into one-hot and feeding them into a neural network and this first layer basically embeds. 

These are two equivalent ways of doing the same thing. We are just going to index since it is much faster. And we are going to discard this interpretation of one hot inputs being inputs into neural nets and just use indexing and embedding tables.

Embedding a single integer like 5 is simple enough. We can just ask PyTorch for the 5th row in `C` or row index five. But how do we simultaneously embed all of these 32 by 3 integers stored in `X` .

Luckily PyTorch indexing is flexible and quite powerful. It doesn’t just work to ask for a single element five like this you can actually index using lists. For example we can get the rows 5, 6, 7 like this

```python
C[[5, 6, 7]]
```

It doesn’t have to be a list but also a tensor of integers. 

```python
C[(5, 6, 7)]
```

This will just work as well. In fact we can also for example repeat row 7 and retrieve it multiple times. That same index will just get embedded multiple times.

```python
C[(5, 6, 7, 7, 7, 7)]
```

So here we are indexing with a one dimensional tensor of integers, but it turns out that you can also index with multidimensional tensors of integers. So we can simply do

```python
C[X]
```

And this just works. The shape of this is [32, 3, 2], [32, 3] is the original shape of `X` and for every one of those 32 by 3 integers we retrieve the embedding vector in [2]. So basically we have that as an example. The code below is an integer for example `1` .

```python
X[13, 2] = tensor(1)
```

If we do `C[X]` and we index into `[13, 2]` of that array:

```python
C[X][13, 2]
```

Then we get the embedding

```python
tensor([1.0015, -0.3502])
```

We can verify that integer one, which was the integer at that location is indeed equal to this

```python
C[1] = tensor([1.0015, -0.3502])
```

So basically PyTorch indexing is awesome and to embed simultaneously all of the integers in x we can simply do `C[X]` and that just works.

## Implementing the hidden layer + internals of torch.Tensor: storage, views

So we have that `W1` are these weights which we will initialize randomly. 

```python
C = torch.randn((27, 2))
emb = C[X]
W1 = torch.randn(())
```

Now the number of inputs to this layer will be [3, 2] because we have two dimensional and we had three of them so the number of inputs is 6. And the number of neurons in this layer is a variable up to us. Lets use 100 neurons as an example. The biases will also be random and we will need 100 of them

```python
W1 = torch.randn((6, 100))
b1 = torch.rand(100)
```

The problem with this is that normally we would take the input in this case that’s embedding and we’d like to multiply it with these weights and then we would add the bias. This is roughly what we want to do. The problem is that these embeddings are stacked up in the dimensions of this input tensor.

```python
emb @ W1 + b1
```

So this matrix multiplication would not work because `emb` has a shape of [32, 3, 2] and we can’t multiply that by (6, 100). So somehow we have to concatenate the inputs together so that we can do this multiplication.

So how do we transform [32, 3, 2] into [32, 6] so that we can actually perform the multiplication.

There are usually many ways of implementing what you’d like to do in torch. Some of them will be faster, better, shorter, etc. That is because torch is a very large library with lots of functions. If we go to the documentation we will see so many functions we can call on these tensors to transform, create, add, them etc. This is kind of like the space of possibility if you will.

One of the things you can do is search for concatenation and find `torch.cat`. This concatenates the given sequence of tensors in a given dimension. These tensors must have the same shape etc. We can use the concatenate operation to in a naive way to concatenate the 3 embeddings for each input.  

We want to take all the examples 0 and then :.

```python
emb[:, 0, :]
```

This plucks out the 32 embeddings of just the first word. The size is [32, 2]. So basically we want the 0, 1, 2 dimensions. And then we want to treat it as a sequence and use `torch.cat` .

```python
emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]
```

It takes a sequence of tensors and then we have to say along which dimension to concatenate. All of them are [32, 2] and we want to concatenate not across not 0 but 1. 

```python
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
```

So this will give as a shape of [32, 6] exactly as we’d like. That basically took 32 and squashed the twos into a six.

Now this is kind of ugly because the code would not generalize if we want to later change the block size. Right now we have 3 inputs, 3 words, but what if we had 5. Then we would have to change the code because we would be indexing incorrectly.

Torch comes to the rescue again because that turns out to be called a function `unbind` . It removes a tensor dimension and returns a tuple with all the given dimensions without it. 

```python
torch.unbind(emb, 1)
```

This gives us a list of tensors exactly equivalent to what we had before. This would give us a `len()`  of 3 and it is the list we had before. We can now call `cat()`  on it.

```python
torch.cat(torch.unbind(emb, 1), 1)
```

The shape is the same [32, 6], but now it doesn’t matter what block size we have and it would work. 

This gives us the opportunity to hint at the internals of `torch.tensor` . Lets create an array of elements from 0 to 17.

```python
a = torch.arrange(18)
```

The shape is just [18]. It turns out that we can very quickly re represent this as different size n-dimentional tensors. We do this by calling `view()` . We can say this is not a single vector of 18, it is a 2 by 9 tensor,  or alternatively a 9 by 2 tensor. Or a 3 by 3 by 2 tensor. As long as the total number of elements multiply to be the same this will just work

```python
a.view(3, 3, 2)
a.view(2, 9)
...
```

In PyTorch `view()` is extremely efficient. The reason for that is something called the **underlying storage**. The storage is just the numbers as a one-dimensional vector. This is how it is represented in the computer memory. When we call `view()`  we are manipulating some of the attributes of the tensor that dictate how the one dimensional sequence is interpreted an n-dimensional tensor. 

What is happening here is that no memory is being changed, copied or created. The storage is identical. When you call `view()`  some of the internal attributes of the view of this tensor are being manipulated and changed. In particular it is something called storage **offset, strides, and shapes** and those are manipulated so that this one-dimensional sequence is seen as different n-dimensional arrays. 

There is a blogpost by Eric called “PyTorch internals” that explains how the `view()` of tensor is represented. This is really like a logical construct of representing the physical memory. It is a pretty good blogpost to go into:

[http://blog.ezyang.com/2019/05/pytorch-internals/](http://blog.ezyang.com/2019/05/pytorch-internals/)

We just have to know that `view` is extremely efficient.

We see that the shape of our `emb`  is [32, 3, 2] but we can simply ask for PyTorch to view this instead as a [32, 6]. The way it gets flattened into a [32, 6] array just happens that these [2] get stacked up in a single row and that is basically the concatenation operation that we are after.

```python
emb.view(32, 6)
```

You can verify that it gets the same result as before. So long story short we can just `view`  as a 32 by 6.

```python
emb.view(32, 6) @ W1 + b1
```

The size is now [32, 100] with the 100 dimensional activations for every of the 32 examples which gives the desired result.

Lets do two things here. Lets not use 32 but use `emb.shape[0]` so that we don’t hardcode and it would work for any block size. If we do `-1` PyTorch will infer what this should be since the `6` wont change.

```python
# these are equal
emb.view(emb.shape[0], 6) @ W1 + b1
emb.view(-1, 6) @ W1 + b1
```

Just to reiterate the `cat` function is much less efficient because it would create a whole new tensor with a whole new storage. There is no way to concatenate tensors just by the view attributes. So new memory would be created.

We also want to take the `tanh` of the tensor.

```python
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
```

These are now numbers between -1 and 1 and we have the shape of [32, 100]. This is basically the hidden layer of activations for every of the 32 examples.

We also have to be careful with the `+` and how broadcasting will work. The shape of `emb.view(-1, 6) @ W1` is [32, 100] and `b1` shape is [100]. 

We have the [32, 100] broadcasting to [100]. Broadcasting will stretch a row of [1, 100] for 32 rows. In this case the correct thing will be happening since the same bias vector will be added to all the rows. It’s always good practice to make sure.

## Implementing the output layer

Lets create `W2` and `b2`.

```python
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
```

The input now is 100 and the output number will be for us 27 because we have 27 possible characters that come next. The biases will be 27 as well. 

## Implementing negative log likelihood loss

The logits which are the outputs of the neural net are going to be 

```python
logits = h @ W2 + b2
```

The shape is [32, 27]

We want to take these logits and exponentiate them to get our fake counts and then normalize them into probability. Exactly like the previous video.

```python
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
```

`prob.shape` is 32 by 27. Every row of prob sums to one, so it is normalized. 

Now we of course we have the actual letter that comes next. That comes from the array `Y` which we created during the dataset creation. It is the identity of the next character of the sequence that we would like to predict. 

We’d like to index into the rows of prob and each in each row we want to pluck out the probability assigned to the correct character.

This is like an iterator of numbers from 0 to 31.

```python
torch.arrange(32)
```

We can index into `prob` in the following way.

```python
prob[torch.arrange(32), Y]
```

This gives the current probabilities as assigned by this neural network by the current setting of the weights. We now have to tune the weights so that they correctly predict the next character.

We can create the loss function in the following way using negative log likelihood loss function:

```python
loss = -prob[torch.arrange(32), Y].log().mean()
```

This is the loss we want to minimize to get the network to predict the correct character in the sequence.

## Summary of the full network

```python
X.shape, Y.shape # dataset
```

(torch.Size([32, 3]), torch.Size([32]))

```python
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
```

```python
sum(p.nelement() for p in parameters) # number of parameters in total
```

Number of elements: 3418

```python
emb = C[X] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
loss = -prob[torch.arrange(32), Y].log().mean()
loss
```

Current loss: 17.7697

## Introducing F.cross_entropy and why

Instead of typing the following code

```python
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
loss = -prob[torch.arrange(32), Y].log().mean()
loss
```

we can use a function already in the PyTorch library. It will calculate the exact same loss.

```python
F.cross_entropy(logits, Y)
```

Current loss: 17.7697. So now the code should look like this

```python
emb = C[X] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Y)
```

When using `cross_entropy` PyTorch wont create all the intermediate tensors, that take up memory. It is also inefficient to run like that. 

### Reason 1

PyTorch will use fused kernels which are clustered mathematical operations. 

The backward pass will also be more efficient analytically and mathematically. It is a much simpler backward pass to implement.

We saw this with micrograd. To calculate `tanh` it was a complicated mathematical expression. But since it was clustered, when we did the backward pass we didn’t individually go propagate through that whole expression we just said it was 1 - t**2 which is a way simpler mathematical expression.

Often the derivative simplifies mathematically and there is much less to implement. 

So not only it is in a fused kernel, but the expressions take a simpler form mathematically.

### Reason 2

Suppose we have the logits of 

```python
logits = torch.tensor((-2, -3, 0, 100))
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
```

In the current example e will get raised to the power of 100 and we will run out of range in the floating point number. Very large logits cannot be passed to `logits.exp()`. 

The way PyTorch solved this they realized that any offset to `logits` + offset will produce the exact same probabilities. Since negative numbers are okay and positive ones are harmful, it counts the maximum number that occurs in the logits and it subtracts it. Therefore the greatest number will become 0 and the float wont overflow to `inf`.

So forward and backward pass will be much more efficient and numerically the logits will be better behaved.

## Implementing the training loop, overfitting one batch

```python
for p in parameters:
	p.requires_grad = True
```

```python
for _ in range(1000):
	# forward pass
	emb = C[X] # (32, 3, 2)
	h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
	logits = h @ W2 + b2 # (32, 27)
	loss = F.cross_entropy(logits, Y)
	# backward pass
	for p in parameters:
		p.grad = None
	loss.backward()
	# update
	for p in parameters
		p.data += -0.1 * p.grad
```

Currently the loss is really good but were overfitting examples on the last 5 words that have 32 examples. It is very easy to make the neural net fit the data.

We are overfitting a single batch of the data.

We wont achieve a loss of zero. The reason for that is that there are multiple examples in the input data that are equal to each other but they yield different labels. But the loss is still quite good.

In cases where there are unique inputs and outputs we are getting good results and overfitting.

## Training on the full dataset, minibatches

Training on the full dataset of 28 000 examples takes a really long amount of time. So it is useful to train on minibatches of the data to improve the time.

We want to randomly select a minibatch and only forward, backward and update on that little minibatch and then iterate on those minibatches.

```python
torch.randint(0, 5, (32,))
```

This will create a tensor with 32 values from 0 to 4. We actually want the shape of the first row here.

```python
torch.randint(0, X.shape[0], (32,))
```

This creates integers that index into the dataset and there are 32 of them. 

```python
for _ in range(1000):
	# minibatch construct
	ix = torch.randint(0, X.shape[0], (32,))
	
	# forward pass
	emb = C[X[ix]] # (32, 3, 2)
	h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
	logits = h @ W2 + b2 # (32, 27)
	loss = F.cross_entropy(logits, Y[ix])
	
	# backward pass
	for p in parameters:
		p.grad = None
	loss.backward()
	
	# update
	for p in parameters
		p.data += -0.1 * p.grad
```

`ix` is used to index into `X` and `Y` . Now we have minibatches that are way faster.

Since we are dealing with minibatches the quality of the gradient is lower so the direction is not as reliable. But the gradient direction is good enough to be useful. It is much better to have an approximate gradient and take more steps then it is to evaluate the exact gradient and take less steps.

We can also check the loss on the entire training set after training on the minibatch.

```python
emb = C[X] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Y)
```

We don’t know if we are stepping too slow or too fast with `0.01` since it was a guess. So how do we determine the learning rate?

## Finding a good initial learning rate

We can try with arbitrary learning rates by eyeing the changes in the loss function and then determine the interval of learning rates that we want to test. 

If the loss function is jumping uncontrollably that means we are overstepping and the model isn’t learning. If the loss is changing too slowly the rate is slow too.

So lets say we determined that a good learning rate is somewhere between -0.001 and -1.

We can use this torch command

```python
torch.linspace(0.001, 1, 1000)
```

This creates 1000 numbers from 0.001 and 1. It doesn’t really make sense to step between them linearnarily so instead create learning rate exponent and instead of 0.001 will be -3 and 1 will be 0. The actual learning rates will be 10 to the power of lre.

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```

We are stepping linearly through the exponents of the learning rates. We are spaced exponentially on the interval.

In the beginning the learning rate will be very low but by the end it will be the max or 1. We are going to step with that learning rate. 

We will also track the loss for each learning rate

```python
lri = []
lossi = []

for i in range(1000):

	# minibatch construct
	ix = torch.randint(0, X.shape[0], (32,))
	
	# forward pass
	emb = C[X[ix]] # (32, 3, 2)
	h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
	logits = h @ W2 + b2 # (32, 27)
	loss = F.cross_entropy(logits, Y[ix])
	
	# backward pass
	for p in parameters:
		p.grad = None
	loss.backward()
	
	# update
	lr = lrs[i]
	for p in parameters
		p.data += -lr * p.grad
		
	# track stats
	lri.append(lre[i])
	lossi.append(loss.item())
```

Now we can plot the learning rates on the x axis and the losses on the y axis

```python
plt.plot(lri, lossi)
```

We might get a graph like this with the exponent of the learning rate.

![Untitled](Building%20makemore%20Part%202%20MLP/Untitled%201.png)

So -1 is a pretty good setting. So 10**-1 is 0.1. Now we can remove the tracking and be confident that out learning rate is decent.

This model performs substantially better than the bigram model already.

When the loss function starts to plato off, we might consider using learning rate decay to slightly improve the model. 

This is a rough example. 

## Splitting up the dataset into train/val/test splits and why

Just because a model has a low loss doesn’t necessarily mean that it performs better. The thing is if we add enough parameters, the model might just memorize the whole training set. It would overfit and perform well only on the set it was given and no new data will be generated. The loss could become very close to zero. 

If you try to evaluate the loss on some withheld names, the loss will become very high.

The standard in the industry is to divide the data into splits:

1. Training split ~80%
2. Dev/validation split ~10%
3. Test split ~10%

The training set is used to get the parameters of the model using gradient descent. 

The dev/validation split is used for development over all the hyperparameters of your model. 

The test split is used to evaluate the performance of the model at the end. We are only evaluating the loss on the test split very sparingly and very few times. Otherwise you risk overfitting to it as well during training.

```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

Here we are dividing the code into training sets after we get the input and label data.

During training we will only train on the training set

```python
for _ in range(30000):
	# minibatch construct
	ix = torch.randint(0, Xtr.shape[0], (32,))
	
	# forward pass
	emb = C[Xtr[ix]] # (32, 3, 2)
	h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
	logits = h @ W2 + b2 # (32, 27)
	loss = F.cross_entropy(logits, Ytr[ix])
	
	# backward pass
	for p in parameters:
		p.grad = None
	loss.backward()
	
	# update
	for p in parameters
		p.data += -0.1 * p.grad
```

Training can take a while and can take multiple days. 

We can then calculate the loss on the dev set which was not used during training and get hopefully a low loss

```python
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
```

If the training loss and dev loss is about the same that means we are not overfitting. 

This model is not powerful enough to memorize the data. 

If the training loss and dev loss are about equal that means we are underfitting. The network is very small and we expect to get performance improvements by scaling up the neural net.

## Experiment: larger hidden layer

Lets increase the size of the net by replacing 100 by 300.

```python
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
```

We now have 10000 parameters not 3000

After this we can train in by iterating lets sat 30000 times and tracking the stats to see results.

During this stage we keep iterating on the training and monitor the situation trying to get the loss to lower. 

Also we can consider changing the context length, tweaking the minibatch, embedding in longer vectors, etc. The goal is to get a better loss.

In a production situation you would initialize multiple experiments with these networks with slightly different parameters, train them and see which one works out the best. 

After training you would test it on the test split and record the number to a paper for showcase.

## Visualizing character embeddings

![Untitled](Building%20makemore%20Part%202%20MLP/Untitled%202.png)

Here we can see the vowels are sort of to the left in a batch, q is on the far right since its rear. We can notice that the model actually learned something about the data and can now generate better looking names.

Jypiter notebook with full code:

[https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)

## Sampling from the model

```python
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
```