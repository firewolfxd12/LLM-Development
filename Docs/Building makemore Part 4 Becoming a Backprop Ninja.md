# Building makemore Part 4: Becoming a Backprop Ninja

Created by: Nick Durbin
Created time: July 27, 2024 3:10 PM
Last edited by: Nick Durbin
Last edited time: July 28, 2024 1:26 AM
Tags: Makemore

## Introduction

So far we made Bigram models and MLPs. We will continue studying MLP to understand backpropagation more. 

## Notes

If the shapes during backpropagation are different, we should be careful. In some cases `.sum` might be used.

### Rules:

- Every time there is a sum in the forward pass, there will be some kind of broadcasting in the backward pass along the same dimension.
- When we have broadcasting in the forward pass, that indicates a variable reuse, which will cause a sum in the backward pass.
- 

### Example:

Lets say we want to manually backpropagate trough this expression and find the gradient expressions of `bndiff` and `bnvar_inv`.

```python
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
```

The sizes are the following

```python
bnraw.shape, bndiff.shape, bnvar_inv.shape
(torch.Size([32, 64]), torch.Size([32, 64]), torch.Size([1, 64]))
```

- To find `dbndiff` we notice that `dbndiff` has to be the same size of `bndiff` .

```python
dbndiff = dbnvar_inv * dhpreact
```

During the multiplication the broadcasting will cause `dbndiff` to be the right shape. Multiplying

- To find `dbnvar_inv` we do a similar calculation

```python
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

Since it will be the wrong shape we have to take the sum.

## Brief digression: bessel’s correction in batchnorm

Using `1/n` is biased and always underestimates the variance. The unbiased version is `1/(n-1)` . When using correction make sure to use it during training and at test time so that there is no training time mismatch.

Full analysis: [Bessel's Correction](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbm5ZamNsTFV0eGk4YzdSd005LUNVQ3JIQWZkUXxBQ3Jtc0ttTjdUZklKSDNhMDkzYjBhVmp3TGFoQXJ3WVFVUGRXcy1WQ28zX2p6MzVvT1JFNzR4WXJCWlkybnlleENiOHVzTXJzSmlCd1V0Z1h0aVROSy1EYXFOejBRMWd4Tjk5OEVRc0ZJajZLakpjbUgtNEhzQQ&q=http%3A%2F%2Fmath.oxford.emory.edu%2Fsite%2Fmath117%2FbesselCorrection%2F&v=q8SA3rM6ckI)

Full code implementation:

[https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)