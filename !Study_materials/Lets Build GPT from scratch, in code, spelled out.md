# Lets Build GPT: from scratch, in code, spelled out

Created by: Nick Durbin
Created time: June 30, 2024 10:35 AM
Last edited by: Nick Durbin
Last edited time: July 31, 2024 4:28 PM
Tags: GPT

# Introduction

- GPT - Generative Pre-trained Transformer
- The model tries to predict the next sequence of characters(or other units) based of another sequence
- OpenAI’s product is highly sophisticated with many iterations of fine tuning which would be really hard to replicate

# Reading and Exploring Data

- The model can only see or emit characters based on its data set
- Tokenization - converting raw text to some sequence of integers according to some vocabulary of possible elements
    - Tokenization involves encoding/decoding text
    - Tokenizing can be done on a character level, word level, or sub-word level
    - Sub-word level is usually used
- Tokenization Schemas:
    - Google uses SentencePiece
    - OpenAI uses tiktoken which is a byte pair encoding tokenizer (BPE)
- Data can be split into train and validation sets
    - Perfect memorization is not needed, only probabilistic generation. Split allows to get a sense of the overfitting

# Data Loader

- Feeding entire text at once is very computationally expensive
- Training a transformer is more effective if chunks are used
    - Chunks are parts of the main data that are picked out to train the data
    - A chunk is basically the maximum context length (also called block)
- A chunk has multiple *examples* of expected data packed into it. The objective is to make a prediction at every position in the chunk.
- A batch is a collection of chunks that the transformer can train on in parallel
- The data loader takes batch_size amount of chunks and then creates a 2d array of the same size that has the correct prediction for each index in the chunks. Batches simply improve computation time and chunks in the same batch don’t affect the actual model in any way.
- The transformer takes a batch_size amount of chunks and looks up the correct integers to predict for every position in the 2d array.
- The batch_size amount of chunks is fed into the neural network.

# Simplest baseline: bigram language model, loss, generation

As we did before we will start with the bigram language model. Using the `torch.nn` library, we will use the `nn.Embedding` module that will return a vector embedding for an index/token. 

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        return logits

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
```

To calculate the loss we will use `F.cross_entropy` function, but in order to use it we have to reshape our tensors so that we can feed them into the function.

```python
if targets is None:
    loss = None
else:
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
	  targets = targets.view(B*T)
		loss = F.cross_entropy(logits, targets)
```

To generate from the model we have to add this function to the Bigram class.

```python
def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

We discussed bigram models in great detail, here is the script for the model:

[bigram.py](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/bigram.py)

# Version 1: averaging past context with for loops, the weakest form of aggregation

In order for the predictions to be better, we have to in some form store information of all previous tokens including the current one. One way to do this is to get the average of the previous tokens. This form is very lossy but it will do for now. This concept is at the core of Transformers.

```python
# consider the following toy example

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
x.shape = [4, 8, 2]
```

One way we could do this is use a for loop (very ineficient)

```python
# We want x[b, t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
		for t in range(T):
				xprev = x[b,:t+1] # (t,C)
				xbow[b, t] = torch.mean(xprev, 0)
```

- bow is short for “bag of words”.

![Untitled](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/Untitled.png)

The code will take a batch and then for the `ith` index in the time vector calculate the average of it and everything before it.

# The trick in self attention: matrix multiply as weighted aggregation

We can be very efficient by using matrix multiplication. 

```python
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
```

![Untitled](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/Untitled%201.png)

In the example above we can see that matrix `c` is storing the sums of the columns from `b` . 

![Untitled](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/Untitled%202.png)

```python
a = torch.tril(torch.ones(3, 3))
```

The trick is the following. If we make `a` a triangular matrix, `c` will now store the sums up to the `ith` position.

```python
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
```

If we divide each row by the sum in `a` we can now get the averages of `b` like we did in the for loop but in a more efficient manner.

![Untitled](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/Untitled%203.png)

# Version 2: using matrix multiply

```python
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
```

# Version 3: using softmax

```python
# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
```

# Positional encoding

We will add a new layer that encodes the positions of the tokens.

```python
def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

```python
# idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arrange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
```

# Version 4: self attention

This is the most important part of the video to understand. We are going to implement a small self attention for a single “head” as the are called.

```python
# version 4: self-attention!
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ x

out.shape
```

Currently our weights are all initialized to 0. The weights for example the vowel “a” wants to attract consonants, so therefore we want information to flow to the consonants in order to make the right prediction. This information is also got from “the past” and this is how self attention solves it. 

Every single token will emit two vectors. It will emit a query and a key.

- Query: What am I looking for?
- Key: What do I contain?

```python
# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
```

The weights should be initialized as the matrix multiplication of `key` and `query` . During this multiplication they communicate with each other and produce a high affinity when needed. 

You also need to add another linear layer for output called `value`. 

```python
# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape
```

In a way `x` is private information to a token. 

# Attention notes

You can think of attention as a directed graph where each token outputs a vector of information to other tokens. the first token only points to itself, the second one points to itself and the first one and by the end the last token points to itself and all of the previous tokens. It is a communication mechanism between the nodes. 

By default none of the nodes have information on their position and we have to encode this position so that they know where they are. In attention you can add a notion of space but it can work without it. 

Also the elements in different batches never talk to each other, they are processed independently from each other.

Currently we have the constraint that future tokens cant talk to the past token, but in the general case that doesn’t have to be the constraint. Sentiment analysis allows for full communication and the attention block it uses is called an encoder block. The one that is currently implemented is called an encoder block, since it decodes language.

There is also cross attention. How is it different from self attention? In self attention all of the keys, queries and values come from the same source. Attention is more general than that. Cross attention is when there is a separate source of information that we want to pull from into our nodes. 

It is also important to divide `wei` by the square root of `head_size` this will make the tensor unit gaussian which is important for the soft max. Otherwise soft max will be pulling information from a single node making it contain less information than needed.

# Multi-headed self-attention

Currently our model only has one attention block. All we have to do is create a new module that has multiple attention blocks. 

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

# Feedforward layer

Currently we didn’t implement the transformer fully. There is still a `feedforward` block that we should implement before we interpret the answer as the logits. 

All it is is a linear layer followed by a non-linearity.

```python
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

# Residual connections

![Untitled](Lets%20Build%20GPT%20from%20scratch,%20in%20code,%20spelled%20out/Untitled%204.png)

Currently we want to implement the block highlighted in light gray. The block basically does communication and computation. Communication is done using multi-headed self attention and computation is done using a feed forward network. 

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

If you go through the training process and go through the block a lot of times, you don’t really get good results. This is because the neural net is quite deep and deep neural nets suffer from optimization issues. 

One thing we should add are residual connections. Residual connections go from the inputs to the targets only via addition. This is useful because addition distributes gradients equally to both of its branches. This creates a residual pathway that communicates directly to the input and the residual blocks don’t contribute anything to the residual pathway at least in the beginning. This improves performance. 

They were already implemented in the code before:

```python
 def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

See more: [01:26:48](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=5208s) residual connections

# Layernorm

Layernorm is similar to Batchnorm. Batchnorm made sure that the outputs were unit gaussian. The only difference is that we are normalizing the rows not the columns. What is different from the original paper is that the layer norms are usually placed below transformations and not above. 

We now have the transformer implemented but it is a decoder only transformer. Now we can start scaling.

# Dropout

Also a dropout module was added that prevents the model from overfitting. What it does it randomly prevents some of the nodes communication during training so that it partially understands a dataset. Then during generation it enables everything, making the model slightly worse at predicting but without overfitting.

# Links

- Google colab for the video: [https://colab.research.google.com/dri...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3JMeGZCLVRoMEV4U0RXOUFyZFpGLTZNZHhDd3xBQ3Jtc0tuT1dhNTlIYXljb0dkZ2t2N3RZTms1Qkw5dG5oWFNVNFE5OFJFZHJMXzNEWmhwOEhnSFE0WjNnN2lsdXJEMFV6V184SUlNVjhUT29DWElBSDdYZEJnM295NTRhVFFpd1VhbEFMLVVsVlVmaTBxeW9qZw&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-%3Fusp%3Dsharing&v=kCc8FmEb1nY)
- GitHub repo for the video: [https://github.com/karpathy/ng-video-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1RkWHYzamU5U2ZtVDk0dVFsV1lPT2RvTkF5UXxBQ3Jtc0tsQlN2M1g0d1RoQW5KNktucjNVdzdDQWNLd3pTcDhWTjdzaTNyc19oaUkxVWh1bjBuTkNIdG5COWltTEJZRk5lREUxem81ZzlBY0RNRU5XajlZSXdqZWlpNERfd2ZnYVhmY2lSdGJIaXUxeTdrbHlqOA&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fng-video-lecture&v=kCc8FmEb1nY)