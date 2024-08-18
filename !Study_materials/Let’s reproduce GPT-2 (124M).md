# Let’s reproduce GPT-2 (124M)

Created by: Nick Durbin
Created time: July 31, 2024 2:42 PM
Last edited by: Nick Durbin
Last edited time: August 1, 2024 6:47 PM
Tags: GPT

## Intro

Today we are going to reproduce the GPT2 model (124M). Loss is calculated on validation data.

## Exploring the GPT-2 (124M) OpenAI checkpoint

The original GPT-2 code is written in tensor flow, which is not used anymore. The hugging face has a version of GPT-2 that is written in PyTorch so it is much easier to load and work with. 

```python
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.frompretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
	print(k, v.shape)
```

The code above gets the state dict with the tensors and their sizes.

The first two layers are the token embedding and the position embedding. Since the positions lookup table has 1024 positions(which is also the context length), this means that each token has 1024 positions that it can be in in the past.

The position embeddings are just a set of weights or floats. If we plot the positions table we will see that it has structure

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled.png)

The positions range from 0 to 1024. It has structure since the positional embeddings end up learning functions similar to cosine or sine that represent every position. This allows the transformer to understand and realize what token goes where and take position into account.

If we look into a the columns, we get a channel and were looking at what that channel is doing as a function of position from 0 to 1023.

These channels respond more or less to different parts of the position spectrum. You can tell that since the curves are jagged, the model wasn’t trained as much. The more you train the model the more smooth it will be.

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%201.png)

In the beginning the weights are started absolutely randomly and it is kind of impressive that they find a pattern during optimization. 

In the original paper the positional encoding table was fixed to specific sinusoids and frequencies. In GPT-2 they are trained from scratch, but they end up recovering these sinusoidal like features. 

If we take a random layer and plot its weights we can notice that it has a specific structure also, which is kind of cool.

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%202.png)

We loaded the weights, and now using the hugging face transformer library we can sample from the model.

```python
from transformers import pipeline, set_seed
generator = pipeline("text-generation", model="gpt2")
set_seed(42)
generator("Hello, I'm  a language model", max_length=10, num_return_sequences=5)
```

Now we want to write our own class and implement it. Lets load the model to our class and see how it works.

# Section 1

## Implementing the GPT-2 nn.Module

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%203.png)

We will implement this transformer that is from the original paper only without the encoder (shown on the right) and the multi-head attention connected to it. 

There are also two main differences. The layer norms are located before not after the mini modules and an additional layer normalization right before the final classifier.

Lets implement the schema used by the hugging face transformers to load the weights. We need something that reflects the hugging face schema.

This can be done with a container `nn.Moduledict` which allows to index to the submodules like a dictionary. `nn.Embedding` can be used also, which is a wrapper for a tensor which allows you to index into its rows.

In the original paper the residual pathway had normalization inside of them, which is not good. You prefer to have a clean residual stream from the input to the outputs. Over time the blocks will affect the outputs but not right away. A clean residual pathway is desirable. 

Another thing is recall that attention is a communication operation. It is where all the tokens are lined up in a sequence and they communicate. It is a weighted sum operation. In an mlp each token is processed individually. There is no information being process in between the tokens. Attention is the reduce and the mlp is the map. Each of the blocks iteratively refines the representation inside the residual stream

The mlp module is basically a GELU layer sandwiched in between two linear layers. GELU is like a RELU but without a flat zero. It has an approximation option, but that is a historical quirk for efficiency. Now there is no real reason to use the approximate version, better just to use the exact one. We are going to reproduce GPT-2 exactly and it used the approximation.

The GELU fixed the problem of RELU giving zero gradient and producing dead neurons, by slightly tilting that region so that it always gives gradients and calculations will not get muted. In Llama this non linearity changes to other ones.

Finally we have the attention operation. In this version we don’t make a separate module for the head but instead code the multihead module at once using pytorch operations. Mathematically it is the same as before.

We have the tokens in a sequence and each token emits three vectors, the query key and value. The queries and the keys multiply each other to get the attention amount. There is also an autoregressive mask. 

It this point we can port over all of the weights from hugging face and sample the model. 

To test that class is correct you can use the frame work and load the hugging face model to get the same results as the pipeline.  

Now we want to not load the weights but train the model from scratch.

## Lets train the model

To train the model we can use the tiny Shakespeare dataset. It is important to initialize the weights to have roughly the same probabilities for every character in the sequence so that it is not confidently wrong.

Simple gradient descent is a decent optimizer, but a more advanced one would be Adam or AdamW. AdamW fixes a bug in the code Adam.

The difference from simple gradient descent is that it has two buffers called the first and the second moment. It is kind of a normalization that optimizes the objective faster than SGD.

The `item()` function in PyTorch ships the value from a tensor to the the cpu if not already.

The `to()` function does not convert objects to a device, instead it returns a pointer to new memory that is on a device. 

```python
buf = buf.to(device) # buf.to(device) by itself doesn't work
```

Using the Shakespeare dataset we are hoping to overfit the data since it is not a lot and get a really low loss. We want to perfectly predict it.

A learning rate of 3e-4 is a decent learning rate during the debugging stage. 

Now we should make a data loader to load fresh batches during the optimization.

## Data loader lite

The data loader needs to make sure to fetch batches from the data set that are exactly `B*T` in a sequential order. Also, we fetch tokens to `B*T+1`  because we need a token for the last token in the current batch. If the data loader runs out of data it should loop back to zero. It will complexify later.

Using the data loader we expect the loss to become larger since it cant overfit a single batch anymore. It shouldn’t be too high because our dataset only has about the third amount of tokens that the model supports. All the other probabilities for characters that are not in the dataset should be driven to zero. 

## Parameter sharing wte and lm_head

When writing the code we introduced a bug. Currently `wte` and `lm_head` have the same tensor shape. 

After inspecting both the tensors have the same `data_ptr()` . What is happening here is that this is a common weight time scheme that comes from the original transformer paper. 

The idea is the following. You want these two matrices to behave similar. If two tokens are very similar semantically, you would expect that they are nearby in the token embedding space. In the same way you would expect them to get the same probabilities at the output of the transformer. Both the bottom and top layers have the notion that similar tokens have similar embeddings. Also the output tensors could be used as input words. 

Basically `wte` is used at the bottom and top of the transformer and the backward pass will get gradient contributions from both branches and add up on the `wte` tensor. So you get a contribution from the classifier layer and then at the end of the transformer you get a contribution at the bottom of it, flowing into the `wte` tensor.  

We also want to control the growth of activations inside the residual stream in the forward pass.

# Section 2

## Lets make it fast

You always want to start with what hardware do you have, what does it offer and are you fully utilizing it.

By default in PyTorch when you create tensors the `dtype` is float32. That is quite a lot and for deep learning that is way too much. The training can tolerate way lower precisions. 19.5 TFLOPS is 1900 floating point operations. The lower the bit per float the more performance you will reach. For training INT8 is not used, but could be for inference. If we bring down the precision we can get a lot more teraflops out of the tensor cores available on the GPUs. When we have lower precision, it is easier to process.  

Not only is there limited storage, but also there is a limit of the speed of which you can access this memory called memory bandwidth. Bandwidth is very important because in some cases you cant load the data fast enough which will cause some tensor cores to be unused. If you are getting 68% utilization you are doing extremely well. Half of the time in a well tuned application the tensor cores are not used because the data is not available. 

## Tensor cores

They are instructions int the A100 architecture. It does a 4 by 4 matrix multiply. There are multiple configurations at what precision the internal accumulate happens and what are the input/output precisions etc. There are a few switches but it is basically a 4 by 4 multiply. And then anytime we have any operations that require matrix multiplication, they get broken up into this instruction of 4 by 4 multiply. Everything gets broken up into this instruction because it is the fastest way to multiply matrices.

Most of the work done in neural nets are just matrix computations. The entire transformer is a bunch of matrix multiplications really. The biggest matrix multiplication by far is the classifier layer at the top.

If there are memory errors:

- Decrease batch size, by dividing by 2
- Enable TF32
- Use BF16 Tensor core
- Add autocast only for fp16
- Use `torch.compile`
- Use flash attention

## float16

FP16 cant represent the range of FP32 numbers. When using FP16 you should also use gradient scalars which complexifies the process. BF16 does not change the code and still works well. 

When using BF16 not all tensors might convert. Some operations will still be in FP32. Functions that are more sussceptible to precision changes like softmax or linear layers will stay in FP32. 

Only some parts of the network are affected by the precision change. 

The code now runs more effectively but were just getting started, there are more optimizations ahead. The better performance isn’t free though. We loose precision but it might make up for itself since we have more runs available. 

## torch.compile

`torch.compile` is some really powerful infrastructure from the PyTorch team. It is a compiler for neural networks and is like gcc for c/c++ code. It is simple to use too.

```python
model = torch.compile(model)
```

It might cost compilation time, but it will cause the code to run much faster. 

Speedup mainly comes from reducing Python overhead and GPU read/writes. The results are really incredible.

Python goes through the code sequentially and doesn’t know what operations are going to happen later.  

`torch.compile` will analyze the your code algorithmically and sees the entire code at the same time. It is able to know what operations you intend to run and optimize that process.

1. It will take the Python interpreter out from the forward pass entirely. It will compile the neural net as a single object with no Python interpreter involved. 

The GPU has a different architecture than the CPU since it has much more cores. They are individually a lot simpler. The GPU like CPU has its own memory also called HBM. 

When the code executes the HBM sends the information to the GPUs cores and this link is extremely expensive and takes a lot of time. Python forces the HBU to dispatch kernels at each mathematic operation constantly using the link and making round trips which is inefficient. PyTorch doesn’t know what operations are going to be later so it can’t optimize this process without `compile`. 

`compile` will see all of the operations in advance and route from HBM to GPU a single time without any round trips. That is one example in whats called kernel fusion and is a major way that makes code speed up. 

Most of the calculations happen in the GPU and it has some memory in it. Most of the memory by far is in the HBM which is a separate chip. On the chip there are streaming multiprocessors (SM), where the calculations happen. An SM has 4 individual quadrants and it has tensor cores, where the matrix multiplication happens. There are also different units for the different types like FP64, FP32 and INT etc. There is memory sprinkled throughout the chip. L2 cache for example lives on the chip and the SMs there are L1 cache and registers. The way the memory is stored is very different than the HBM in terms of how the silicon looks like. 

There is memory inside the chip but there is not a lot of memory. Operator fusion allows to do the calculations on the chip before writing it back to the HBM. 

## Flash attention

`torch.compile` is amazing but there are operations that `torch.compile` will not find. An example of this is flash attention which is an algorithm for performing attention and running it a lot faster. 

Flash attention is a kernel fusion operation, but it is a fusion operation that `torch.compile` cannot find. Flash attention does more flops, but is still significantly faster. It is very mindful of the memory hierarchy. It mindful about what is in the high bandwidth memory, the shared memory and it is very careful with how it orchestrates computation such that it has fewer reads and writes to the high bandwidth memory. Flash attention does not read and write the large N by N attention matrix to HBM, resulting in an 7.6x speedup on the attention computation. 

The algorithm relies on a softmax trick that shows how you can incrementally evaluate the softmax without having to realize all of the inputs for the normilzation. You do this by having intermediate variable N and L and there is an update to them that allows you to evaluate the softmax ona inline manner. Flash attention 2 also came out. 

Flash attention is based on a softmax trick that came out of Nvidia in 2018. Later Stanford used it in Flash Attention only 4 years later. They realized that they can use the trick with the Softmax calculation into a single fused kernel called Flash Attention. 

In PyTorch this optimization can be called using the following line:

```python
y = F.scaled_dot_product_attention(q, k, v, is_casual=True)
```

 Its the same computation and we went from 130ms to 96ms.

## Nice/ugly numbers

So there are numbers that are nice and some are ugly. For example 64 is a nice number, 128 is even nicer, 256 is beautiful. What makes these numbers beautiful is that you can divide them by 2 many times. 13 and 17 are examples of ugly numbers. 

Everything in cuda works in powers of two and lots of kernels are written in terms of power of 2. When inputs are not made of nice numbers there is handling involved that makes the time longer. 

In our code, the vocab size, layers in gpt2-xl, etc. are ugly numbers. When you fix these things you should increase the number until its the nearest power of 2. In our case we should change 50257 to 50304. We are going to be doing more flops but the time changes from 96ms to 93ms. 

It is useful to pad your inputs to use powers of two. 

# Section 3

## Hyperparameters, AdamW, gradient clipping

By this point we improved performance by 11x. Now we should do algorithmic improvements. The GPT2 paper doesn’t really tell much about the optimization process so for that we should turn to the GPT3 paper. Int the appendix they have more hyperparameters and they have more detail on how models were trained, but the models were never open source. The GPT2 and GPT3 architectures are very similar but the context length was increased from 1024 to 2048 and some hyper parameters were tweaked. That was the major change. Other than that they are basically the same model, but the GPT3 was trained a lot longer with more thorough evaluations. The GPT3 model is 175B instead of 1.6B in the GPT2. 

The GPT3 was trained with Adam3 with beta1 = 0.9 and beta2 = 0.95. eps = 10e-8. Sometimes the model can get a batch that significantly differs from the original data which can cause a high loss and therefore a high gradient. A lot of people clip the global norm of the gradient at 1.0 so that the model doesn’t become shocked with batches that differ significantly. 

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

The norm should also be recorded, because if it goes up that is bad or if there is a spike. It is also normal for the norm to be high at initial stages since the model is completely random. After the first few stages it usually stabilizes.

There is also a learning rate decay schedule that was implemented. The function was the cosine decay for learning rate down to 10% of its value with warmup. 

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%204.png)

We are not exactly following the paper, because their training horizon is 300B tokens. Their decay time is less than the max steps time, when for us it is exactly equal. 

Deciding what learning rate to use is totally up to you. Cosine learning rates have been popularized by GPT2 and 3. 

## Batch size schedule, weight decay, FusedAdamW, 90ms

There is also a function for batch size. It starts smaller then grows. We are not going to use because it complicates a lot of the arithmetic. It is not a major improvement and only affects speed not the algorithm. 

In the beginning the model is in a very atypical setting and you are mostly learning to ignore the tokens that don’t come up in the training set very often. You are learning very simple biases. Every single example that you put through your network is basically just telling you what token you use which means the network treats every batch roughly the same in the first parts of the optimization. If gradients are pretty much similar why do a batch size of millions. Once you’ve learned all the simple stuff, that is when the work starts. When gradients become more decorrelated per examples then it matters to increase the batch size. We are going to skip this optimization though.

Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. So when you draw a sequence it is gone until the next epoch of training. We are already doing this because the data loader iterates over chunks of data. 

The GPT2 models use weight decay of 0.1 to provide a small amount of regularization. 

## Gradient accumulation

The relationship between weight decay, learning rate, batch size and Adam parameters is a very complicated relationship described in optimization literature. For the most part we are copy pasting the settings that OpenAI used, but this is a complicated topic.

For different models we have different hyperparemeters for the transformer that dictate the size of the transformer network. We also see that batch size grows with the size of the model.

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%205.png)

We cant just use 0.5M batch size for our model because the GPU wont handle it. We still want to use this batch size because it is correlated with all the other optimization hyper parameters and the learning rates and so on. We want to have a faithful representation of all the hyper parameters and use a batch size of 0.5M. So how are we going to use 0.5M if our GPU doesn’t support it. For that we will use gradient accumulation to simulate in a serial way any arbitrary batch size that we set. We will have to process multiple sequences and add up all the gradients from them to simulate a batch size of 0.5M.

In our case we will do the forward and backward pass 32 times and a single update.

```python
for micro_step in range(grad_accum_steps):
		...
		loss.backward()
```

The `loss.backward` always does `+=` on the gradients so they will accumulate. 

## Distributed data parallel (DDP)

It works in a simple way. If we have 8 GPUs we are going to launch 8 processes. For each process the training loop is going to look the same. Each GPU is working on the same code that we wrote, but secretly they are going to be working side by side on slightly different parts of the data. You have to add one more part where you are supposed to calculate the average of the gradients from the GPUs. 

To use all 8 of them we are not just going to launch the code using `python train_gpt2.py`. We are going to be running it by a special command called torch run. Torch run when it runs the scripts will make sure to run the 8 GPUs in parallel and it created environmental variables, where each of the processes can look up which of the processes it is. The only difference between these processes is that they all have a different `ddp_rank` from 0 to 7. In a multinode setting you would also use `ddp_local_rank` . Local rank is the rank of the GPU on a single node. The `ddp_world_size` is total amount of GPUs you have.

```python
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

In the forward pass everything should be identical, but during the backward pass using the simple setting once the backward pass is over on each GPU, each independent GPU has the gradient for all the parameters. What DDP will do once the backward pass is over it will call `all_reduce` and it basically does an average across all the ranks of their gradients. It will then deposit the average on every single rank. It just synchronizes and averages the gradients. 

DDP is more involved than that because it can dispatch communications for the gradient while the backward pass is still happening. There is overlap in the calculating if the gradients and the synchronization of them and the backward pass.

In order for the code to average the gradients and be efficient at it we have to change the code. The goal is to get the average gradients and average loss for all the processes. 

## Datasets used in GPT-2, GPT-3, FineWeb (EDU)

Lets see what datasets were used in the GPTs. The training dataset for GPT2 wasn’t released. GPT2 used outbound links from reddit. In GPT3 they used CommonCrawl. CommonCrawl is known to have data that is absolutely random which might now be desirable for your model. It does have good stuff but there is ad spam etc.  

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%206.png)

One nice dataset is FineWeb which uses CommonCrawl but filters it to get up to (15-trillion tokens). Hugging face also released a FineWeb Edu subset of FineWeb with educational content. 

For our purpose we will use the `sample-10BT` version since it achieves performance similar to the original GPT 2. We will download it, process it, and make sure our data loader can work with it. The filter that FineWeb Edu used was Llama 3 70B which judged the content to go through the filter. 

We created a `fineweb.py` script that loads data into shards which stores data in a numpy array which is similar to torch tensors. This means that we should update the `DataLoaderLite` to iterate over the shards. 

## Validation data split, validation loss, sampling revive

Having a validation split is important to make sure you are not overfitting. 

## Evaluation: HellaSwag, starting the run

While it is training we will use one evaluation that will be used to supplement the validation set. 

The way HellaSwag works, it is basically a sentence completion dataset. It is multiple choice where there is a shared context and you have to select the sentence continuation that makes sense. Only one of them are a natural continuation of the sentence. Models that are not trained well wont be able to complete it. 

It covers a wide range of topics samples from the internet. The wrong options are deliberately adversarially sourced. LLMs find them difficult but humans find them easy. Humans have 95%> accuracy but models have an about 40%< at the time (5 yrs ago). The dataset has been solved an models can achieve a 96% accuracy rate. Back then they would only get 50%. 

The reason people like is that it is a smooth eval. Smooth eval offers random signal which means that smaller models will start at the random chance of 25%, but they will slowly improve. 

Small models can’t actually do multiple choice so we have to give it to them in native form. Native form is a token completion. 

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%207.png)

We construct a batch of 4 rows and T tokens. There is shared context and only one of the options is correct. Only the longest option gets used while the options will have padded dimensions. 

The way the LLM predicts is that we feed it the option and look at the highest average probability for the tokens. That will be the most likely completion for the LLM.

Some evals might not do it this way, doing it in a multiple choice format. Our model cant do that because of its size. We will feed it one option at a time and the correct option has to win out. 

# Section 4

## Results in the morning! GPT-2, GPT-3 repro

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%208.png)

After training the model we see that we surpassed GPT2 124M model on the HellaSwag evaluation. We only trained on 10B tokens while GPT2 trained on 100B tokens. There might be multiple reasons for this:

1. OpenAI GPT2 was trained on a much wider data distribution.
    1. For us FineWeb Edu is only in English and there is not that much math and code
    2. Multilingual, math, and code could have been stealing capacity from the GPT2 model
2. HellaSwag eval is very old. Some of it might have been in the training set for FineWeb
    1. If that is true then we would be looking at the training curve and not the validation curve
3. It is possible that the tokens we used are more high quality
    1. The OpenAI GPT was created a long time ago in LLMs, where good practices weren’t really around for data sets

We have some confidence that we are not doing something completely wrong. It is probably the case that when people create these sets of data, they try not to include data from evaluation sets.

After re running the the training with 4 epochs we almost made it to GPT3 accuracy but not quite.

![Untitled](Let%E2%80%99s%20reproduce%20GPT-2%20(124M)/Untitled%209.png)

You can notice that the graph on the left has sharp edges about 4 times. This is most likely due to the data loader loading data sequentially in the shards, not by random offsets. That is one improvement you could do. 

Also the learning rates from the GPT3 paper are extremely conservative and you can get away with faster training. A lot of the hyper parameters are quite tunable so that could boost performance. 

You could basically make the GPT3 model by just slightly changing the batch size and context length.

You might also consider using more evaluations like lm-evaluation-harness to evaluate your model.

To make an SFT model you would just swap the data for user and assistant data and add you fill in the user tokens and sample the assistant tokens.

## Shoutout to llm.c, equivalent but faster code in raw C/CUDA

It turns out that if you write the code in C it runs faster and gets an equivalent result.

If you are only interested in training a GPT2 or GPT3. llm.c is:

1. Very fast
2. Takes less space
3. Faster to start
4. Faster per step

## Summary, phew, build-nanogpt github repo

- We were looking at the GPT2 and GPT3 papers
- We were looking at how to setup training runs and all the considerations involved
- We wrote everything from scratch
- Over a 2 hr training run or an overnight run we can match the 124M parameter checkpoints of GPT2 and GPT3 to a very large extent
- The code that we wrote can train larger models if you have the computing resources