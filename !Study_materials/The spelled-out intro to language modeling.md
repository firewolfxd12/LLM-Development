# The spelled-out intro to language modeling: building makemore

Created by: Nick Durbin
Created time: July 23, 2024 2:43 PM
Last edited by: Jovan Plavsic
Last edited time: July 29, 2024 6:20 PM
Tags: Makemore

# Intro

- Makemore makes more things that you give it
- Makemore is trained on a dataset then it can create new data that looks similar to the original information
- It is a character level language model
- Many language model neural nets are implemented including:
    - Bigram (one character simply predicts a next one with a lookup table of counts)
    - Bag of Words
    - MLP - Multi layer perceptron
    - RNN
    - GRU
    - Transformer - an equivalent one that GPT 2 has which is a modern NN, but on a level of characters
- In a character level language model the next character is predicted based off of the characters before it. Each name in a dataset is an example of how the model should output.
- For this example we will try to make a model that outputs a new name based off of dataset of names it was given

# Bigram language model

- Always working with 2 characters at a time
    - We only look at one character we are given and are trying to predict the next character in a sequence
    - Only models a small local structure
    - Even if more info is given we only look at the previous character to predict the next one
- Python code that counts the amount of certain pairs of characters using a dictionary

```python
b = {}
for w in words:
	chs = ['<S>'] + [w] + ['<E>']
	for ch1, ch2 in zip(chs, chs[1:]):
		bigram = (ch1, ch2)
		# b.get(bigram, 0) + 1 - counts up all the bigrams
		b[bigram] = b.get(bigram, 0) + 1
```

- It is more efficient to use a 2D tensor in the PyTorch library. 27 by 27 since there are 26 letters in the alphabet and a special ‘.’ character
- Same code but with torch library:

```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
	chs = ['.'] + [w] + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		N(ix1, ix2) += 1
```

- The 2D array N has all the information necessary to sample from the bigram character level language model

# How sampling works in PyTorch

In order to sample it is necessary to convert the counts into probabilities

```python
g = torch.Generator.manual_seed(2147483647)
p = torch.rand(3, generator = 3)
p = p/p.sum()
```

g is a generator in order to get the same results on different computers

p is a tensor with three random values using the generator

p/p.sum() converts all of the numbers in the tensor to a value from 0 to 1 to get a probability

```python
torch.multinomial(
p, 
num_samples=20, 
replacement=True, 
generator=g
)
```

The code above creates a tensor using the probabilities from p to create a num amount of samples.

Example:

```python
p = tensor([0.6064, 0.3033, 0.0903])
res = torch.multinomial(
p, 
num_samples=20, 
replacement=True, 
generator=g
)
res = tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]])
```

In the res tensor the probability to randomly select a zero is 0.6064, one 0.3033, and two 0.0903. The probabilities from p determine the chance of its index being randomly selected. p is the distribution in the sample

# Applying sampling to makemore

We set p to the first row (or the row that has the counts of the characters after the special start character ‘.’)

```python
p = N[0].float()
```

Then we convert the counts into probabilities by dividing each count by the sum of the row

```python
p = p/p.sum()
```

Using the distribution p we can now sample a single number based off of the probabilities in the tensor

```python
g = torch.Generator.manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
```

`itos[ix]`  will convert the number into the corresponding character after which we can repeat the process on this new row. We repeat until we reach the `.`  character after which we break out of the loop

## Building the loop

```python
g = torch.Generator.manual_seed(2147483647)

ix = 0
while True:
	p = N[ix].float()
	p = p / p.sum()
	ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
	out.append(itos[ix])
	if ix == 0:
		break
```

The code above will generate a single word that looks like a name. The Bigram model isn’t really good so results may be pretty bad. If we want to generate more names we can just put the whole thing in a loop.

For efficiency we can convert the counts matrix into a probability matrix to not recount the row every time inside the loop.

```python
P = N.float()
P = P / P.sum(1, keepdim=True)
```

The `1` seen in the argument of the `sum` function specifies the axis. In this case it is taking the row (column is 0).

`keepdim`  makes sure the tensor that the sum returns is a column [27, 1] so that during broadcasting the column will turn into a square matrix so that each row will have the right corresponding sum amount.

If `keepdim` was `False` the sum function would return a row [27] and during broadcasting [1, 27] each row will be broken.

See more about broadcasting:

[The PyTorch library](The%20PyTorch%20library.md)

# After changes:

```python
P = N.float()
P = P / P.sum(1, keepdim=True)
g = torch.Generator.manual_seed(2147483647)

for _ in range(5):

	out = []
	ix = 0
	while True:
		p = P[ix]
		ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
		out.append(itos[ix])
		if ix == 0:
			break
```

Changes made:

1. The code will generate 5 names due to the new `for` loop
2. The probabilities wont be recalculated on each iteration since we have a new matrix `P` with everything already set up for maximum efficiency.

# Loss function (like micrograd)

In order to measure the quality of our bigram model we have to assign it a number that takes the probabilities of the bigrams into account.

`The Likelihood` is what is typically used and it is equal to the product of all the bigram probabilities. A good model has a really high product of these probabilities.

The bigram model will have a really low likelihood since the probabilities multiplied by each other will return an even smaller number. Because of this, most people use the `log likelihood`  for convenience.

The `log` function has the following property:

```python
log(a*b*c) = log(a) + log(b) + log(c)
```

In out code we can simply accumulate all the logs and find the `log likelihood` 

```python
log_likelihood = 0.0
for w in words:
	chs = ['.'] + [w] + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		prob = P[ix1, ix2]
		logprob = torch.log(prob)
		log_likelihood += logprob
```

The loss function has the semantics that “low is good”. To get the loss function we have to negate the `log_likelihood` .

```python
nll = -log_likelihood
```

Using this loss function we can determine the quality of the model. The higher the number the worse the predictions are.

A nice optimization is to get the average log likelihood by dividing `nll` by a counter of bigrams.

### Model smoothing

We can now determine how likely a model would predict a certain name using the following code

```python
log_likelihood = 0.0
n = 0

for w in "andrejq":
	chs = ['.'] + [w] + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		prob = P[ix1, ix2]
		logprob = torch.log(prob)
		log_likelihood += logprob
		n += 1
		
nll = -log_likelihood
```

If we execute the code above the model will return `nll` to be equal to `inf` . This happens because the bigram “jq” in “andrej” has a `-inf` chance of being generated. 

To fix this we can smooth out the model by adding a count of one to the whole probability table.

```python
P = (N+1.float)
P /= P.sum(1, keepdims=True)
```

The more we add the more uniform the model will be and the less you add the more peaked it will be.

# Summary so far

- We have trained a respectable bigram character level language model by:
    - Looking at the counts of the bigrams and normalizing the rows to get probability distributions
- We can use the parameters of the model to perform sampling of new words
- We can evaluate the quality of the model by using the loss function
    - It is giving high probabilities to the actual next characters in all the bigrams in the training set

# The Neural Network Approach

We will end up in a similar position but the approach will be different. We will cast the problem of bigram character level language modeling into the neural network framework. 

The NN is still a bigram character language model. It receives a single character as an input and using the weights it is going to output the probability distribution over the next character in a sequence.

We will be able to evaluate any setting of the parameters of the neural net because of the loss function.

We are going to look at the probability distribution and tune the network.

Gradient based optimization will be used and since we have the loss function we will be able to tune the weights to correctly predict the probability distribution of the next character. 

# Creating the bigram data set for the neural net

Create the training set of bigrams (x, y)

```python
xs, ys = [], []

for w in words:
	chs = ['.'] + [w] + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		xs.append(ix1)
		ys.append(ix2)
		
xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

xs is a tensor with [0, 5, 13, 13, 1] and 

ys is a tensor with [5, 13, 13, 1, 0]

We are essentially labeling the data meaning 0 should return 5, 5 should return 13, 13 should return 13 and so on…

<aside>
💡 `torch.tensor` and `torch.Tensor` are separate classes that can be constructed.
 The difference is the `.dtype` . `tensor` is set to int64, while `Tensor`  is set to float32 by default

</aside>

# Feeding ints into neural nets: one hot encoding

One hot encoding is a common way to encode integers for NN’s. All it does is assign a vector of zeros with a one at the index that is equal to the number.

**One hot encoding:**

5 → [00001…00]

3 → [00100…00]

Using PyTorch:

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes = 27).float()
```

`xenc` is now a tensor with the numbers converted to encoded vectors

# The “neural net” with one layer

To initialize the weights `torch.randn` can be used which returns a tensor with random numbers drawn from a normal distribution. 

In normal distribution most of the values are around 0 and it is really rare for a number to be larger than 3.

![Untitled](The%20spelled-out%20intro%20to%20language%20modeling/Untitled.png)

We can create a single neuron with 27 inputs

```python
W = torch.randn((27, 1))
```

To create a layer with 27 neurons we can just change the 1 to a 27 after which we can multiply the current weights by the values in `xenc` .

```python
W = torch.randn((27, 27))
xenc @ W
```

*@ - is a matrix multiplication operator that acts exactly the same as matrix multiplication in math*

The result of `xenc @ W` is telling us the firing rate of every neuron based on the encoded data it received.

This means that if the element `(xenc @ W)[3, 13]`  has a value of 0.5, then it is giving us the firing rate of the `13` neuron looking at the `3` input in `xenc` . 

The result was achieved by the dot product of the `13` column in the weights and the `3` element in `xenc` . Using matrix multiplication is just an efficient way of counting this dot product since it happens in parallel also.

# Transforming neural net outputs into probabilities: the soft max

At this point we have 27 inputs and 27 neurons in a single layer. These neurons perform `W @ xenc` , they don’t have a bias, and they don’t have a non linearity like tanh. We are going to leave the neurons to be a linear layer.

In the Bigram model we counted the bigrams and then divided by the sums to get probabilities. We want to get something similar from the NN. At the moment we only have a set of weights with negative and positive numbers from the normal distribution. The goal is to get the weights to represent the probabilities for the next character. 

Probabilities have a special structure: they are positive numbers and they sum to one which is impossible to get from a NN. 

The output can’t be counts since the counts are positive and they are integers which are not a good thing to output from a net also.

Instead we are going to make the NN output the log counts. To get the counts we are going to get the log counts and exponentiate them.

### Exponentiation

Exponentiation `e^x` takes numbers that are negative or positive. If we plug in negative numbers the output will always be below one and positives will output greater than one.

![Untitled](The%20spelled-out%20intro%20to%20language%20modeling/Untitled%201.png)

```python
(xenc @ W).exp()
```

The result of the code above will turn all the negative numbers into positive ones, and the positive ones into even more positive ones.

The exponentiated results can be interpreted as the counts similar to the ones in the Bigram Model. 

The exponentiation makes the results positive and they can take on various values depending on the settings of `W` . 

```python
logits = xenc @ W # log counts also called logits
counts = logits.exp() # equivalent to N
prob = counts / counts.sum(1, keepdims=True)
```

Every row will now sum to one, since it is normalized

Basically the net accepts a letter encoded in a vector using one shot encoding. We multiply this vector by the values of the weights of each neuron (dot product). Eventually in the counts matrix we will get a single value that is the result of the dot product. That we then interpret as the probability of the next letter in the sequence using exponentiation.

All of these operations are differentiable which means we can backpropagate through them. We are getting out probability distributions for a specific letter.

The goal now is to tune the weights of the network such that it returns the probabilities of the next letter in the sequence. Tuning can be done using backpropogation which uses the loss function.

# Summary, preview to next steps, reference to micrograd

1. We have an input data set `xs` .
2. For the input data we have labels `ys` for the correct next character in a sequence.
3. Then we randomly initialized 27 neurons’ weights, each neuron receives 27 inputs. 

```python
g = torch.Generator().manual_seed(2147483647)
w = torch.randn((27, 27), generator=g)
```

1. Input to the network is one hot encoding

```python
xenc = F.one_hot(xs, num_classes=27).float()
```

1. Predict log counts

```python
logits = xenc @ W
```

1. Counts tensor equivalent to N in the Bigram Model and interpret probabilities (AKA Softmax)

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
```

Softmax is an extra layer in an NN that takes in an output and then converts it into a probability like structure:

- All results are from [0-1]
- The results in a row add up to one

# Vectorized loss

Since were doing classification negative log likelihood is effective to use for loss. In the binary categorizer we used regression that used a different loss function.

To calculate loss we use the average negative log likelihood.

To do this we have to look at the probability the network assigns to the actual next character in the bigram. 

To efficiently access these values we can once again use the PyTorch library

```python
probs[torch.arange[5], ys]
```

- `torch.arrange` returns a tensor([0, 1, 2, 3, 4]).
- `ys` stores the label for the actual next character in the sequence.
- `probs` is the 2D array that stores the probabilities of what the network assigns to each letter, we are interested in the one that is in `ys` .

Calculate the negative log likelihood aka the loss in vectorized form

```python
loss = -probs[torch.arange[5], ys].log().mean()
```

# Backward and update in PyTorch

Currently our code looks like this

```python
# randomly initialize 27 neurons' weights. each neuron recieves 27 inputs
g = torch.Generator().manual_seed(2147483647)
w = torch.randn((27, 27), generator=g, requires_grad=True)

# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts,  equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange[5], ys].log().mean()
```

PyTorch keeps track of all the operations under the hood making a graph that represents the mathematical expression for the loss function. 

This graph/DAG is used in`loss.backward()` to calculate the gradients of all weights and encoded letters.

```python
# backward pass
W.grad = None # set gradient to zero
loss.backward()
```

`W.grad` after the `loss.backward()` function is called is filed up with values that show how much influence a specific weight has on the loss function. 

This gradient information is then used to update the weights of the NN:

```python
W.data += -0.1 * W.grad
```

Nudging the `W.data` slightly against the direction of the gradient by `0.1` will eventually improve our loss hence making a slightly better model on one iteration. This process is called gradient descent.

# Putting everything together

```python
xs, ys = [], []
for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		xs.append(ix1)
		ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)

# initialize the "network"
g = torch.Generator().manual_seed(2147483647)
w = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(200):
	
	# forward pass
	xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
	logits = xenc @ W # predict log-counts
	counts = logits.exp() # counts, equivalent to N
	probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
	loss = -probs[torch.arange(num), ys].log().mean()
	
	# backward pass
	W.grad = None # set to zero gradient
	loss.backward()
	
	# update
	W.data += -50 * W.grad
```

In the beginning of the video we had a Bigram model that just counted the probabilities. Our loss was roughly 2.47 after smoothing. Before smoothing it was roughly 2.45. 

Using the NN we achieve the same result but with gradient based optimization. We got to 2.46.  

Since the bigram model is so simple we were able just to count the probabilities through a brute force approach. 

The NN approach is significantly more flexible. We can now expand this approach and complexify the NN. 

Currently we are feeding a single character into the NN and the output is extremely simple, but we are about to iterate on this substantially. We will be taking multiple characters and feeding them into more complex neural nets. Fundamentally the output of the NN will always be logits. Those logits will go through the exact same transformation:

1. They will go through a soft max
2. Calculate the loss function based on negative log likelihood
3. And optimize

As we complexify the NN and work all the way up to transformers the approach will change in a small way. The only thing that will change is the way we will do the forward pass where we take the previous characters and calculate the logits for the next character in a sequence. That will become more complex, but we will use the same machinery to optimize it. 

It is not obvious that we would extend the bigram approach into a case where there are more characters in the input. In the old approach the tables would get way to large since there are too many combinations of what previous characters can be. 

If we have one previous character we can keep everything in a table - the counts, but if you have the last 10 characters in the input, we cant keep everything in the table anymore. The bigram model is an unscalable approach, while the NN is significantly more scalable and can be improved on overtime. 

Thats what we will be doing next.

# Note 1

`xenc` was made out of one-hot vectors. The one-hot vectors were multiplied by the `W` matrix. We think of this as multiple neurons being forwarded in a fully connected manner. 

Actually what is happening if you have for a one-hot vector with a 1 at the 5th dimension, then because of the way matrix multiplication works, multiplying that one-hot vector with `W` actually ends up plucking out the 5th row of `W` .

This causes `logits = xenc @ W`  to just become the 5th row of `W` (since the rest just become equal to zero due to matrix multiplication). 

This is exactly what happened before in the bigram model. We took the first character and then that first character indexed into a row, and that row gave the probability distribution for the next character. The first character was used as a lookup to get the probability distribution. 

The same happens in the NN. `logits` becomes the appropriate row in `W` . So `W` essentially becomes reflects the `N` array of counts from before. But `W.exp()` is technically the actual array. 

The `N` array of counts was populated by counting the bigrams, whereas in the gradient based framework we initialize the array randomly and let the loss guide us to arrive at the exact same array. 

# Note 2

Remember the smoothing where we added fake counts in order to smooth out and make it more uniform the distributions of these probabilities which prevented us to assign 0 probability to any one bigram. 

```python
P = (N+1).float()
P /= P.sum(1, keepdims=True)
```

The more we add to `N` the more uniform/even the probability will be. It turns out that the gradient based framework has an equivalent to smoothing.

In particular think of the `W` which we initialized randomly. We can also think about setting all the `W` to zero. If all of them are zero, then the logits will become all zero, the exp will become all one and the probabilities become exactly uniform. 

If the `W` are equal to each other, the probabilities come out completely uniform. Trying to incentivize `W` to be near zero is equivalent to label smoothing. The more we incentivize this in a loss function, the more smooth distribution we will achieve. 

This brings us to regularization. We can augment the loss function to have a small component that is called the regularization loss. 

```python
(W**2).mean()
```

There will be no signs since we are squaring. We will achieve zero loss if W is zero, but if W has non zero numbers, you accumulate loss. 

We can add this to the forward pass for a better loss function

```python
...
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(num), ys].log().mean()
			+ 0.01 * (W**2).mean()
...
```

This optimization adds two components. Not only is it trying to make all the probabilities work out, but there is also a component that tries to make all the probabilities be zero. You can think of this of like adding a spring force that tries to push W to be zero. 

The strength of the regularization is exactly controlling the amount of counts that we added before. Adding more counts corresponds to increasing `0.01` . 

# Sampling from the neural net

Old Bigram model sampling:

```python
g = torch.Generator.manual_seed(2147483647)

for _ in range(5):

	out = []
	ix = 0
	while True:
		p = P[ix]
		ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
		out.append(itos[ix])
		if ix == 0:
			break
```

Sampling from NN:

```python
g = torch.Generator.manual_seed(2147483647)

for _ in range(5):

	out = []
	ix = 0
	while True:
		xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() # input to the network: one-hot encoding
		logits = xenc @ W # predict log-counts
		counts = logits.exp() # counts, equivalent to N
		p = counts / counts.sum(1, keepims=True)
	
		ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
		out.append(itos[ix])
		if ix == 0:
			break
```

Both code snippets will output the exact same result. These models are identical. They achieve the same loss and `W` is the `N.log()` from the old model. But the NN is more scalable.

# Conclusion

1. Introduced the Bigram character level language model
2. Saw how we can train, sample, and evaluate quality of the model using negative log likelihood loss
3. We trained the model in two completely different ways that actually give the same result.
    1. The first way we counted out the frequency of the bigrams
    2. The second way we used the negative log likelihood loss as a guide to optimize the counts array so that the loss is minimized in a gradient based framework
    3. Both models give the same result
4. The gradient based framework is really flexible. Currently we are taking the single previous character, and taking it through a single NN layer to calculate the logits. This is about to comlplexify.
5. In the following videos we will be taking more and more of the characters and feeding them into a NN. The NN will output logits. The process will stay identical. 

*The NN will now complexify all the way to tranformers…*

Thats going to be pretty awesome and I’m looking forward for it and for now bye. 👋