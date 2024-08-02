# The PyTorch library

Created by: Nick Durbin
Created time: July 23, 2024 4:51 PM
Last edited by: Nick Durbin
Last edited time: July 28, 2024 8:00 PM
Tags: PyTorch

PyTorch is an open source machine learning framework that accelerates the path from research prototyping to production deployment.

# Useful tutorials

- Python + Numpy tutorial from CS231n: [https://cs231n.github.io/python-numpy...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGZqby1LRlZQSG5IM0NnZkhBWmpFNDZya2Fld3xBQ3Jtc0tuQnhpMGdEU0dWUDk5MnpxZW85RGJmV2JLZ3o4dWlqV180c0k5NHZnN1BibjRKS095TExRQXBZUDF3b2l6enlibUVUcVVLU2JKdnQtSzNpM1RKS0k2aWtra2ZyV1paZTIzRk14RUdjVzVtcko1UndfMA&q=https%3A%2F%2Fcs231n.github.io%2Fpython-numpy-tutorial%2F&v=PaCmpygFfXo)  

We use torch.tensor instead of numpy.array in this video. Their design (e.g. broadcasting, data types, etc.) is so similar that practicing one is basically practicing the other, just be careful with some of the APIs - how various functions are named, what arguments they take, etc. - these details can vary.
- PyTorch tutorial on Tensor: [https://pytorch.org/tutorials/beginne...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFpkUVZVLWRlSnhjRjIwUTc3Z2xnd2xKcVp2UXxBQ3Jtc0tsNWoyeWhKZGJLSXRValVrV3RfZW41Nl9RWW1ILW9TUGJYTWc3OF9iajJETVBXZWp3LTROeWRqd05sT2hGWUdxMEZ3T0ZXTUdwTUNWeUQzdXh1TmJtaXJPR3VHdXNTMFAtWG1nNjFZWDZObGgzamhyVQ&q=https%3A%2F%2Fpytorch.org%2Ftutorials%2Fbeginner%2Fbasics%2Ftensorqs_tutorial.html&v=PaCmpygFfXo)

- Another PyTorch intro to Tensor: [https://pytorch.org/tutorials/beginne...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGcwSG1GWmhWTzBLcUZxY2NURzJCeVV5OE94UXxBQ3Jtc0tsVjFNZV9HRWt0aU5vWFNBQjAxR3ZLdk9BeGJ3NjZjTVNlUEhxeVhVQ2lqTHcwaXluWjk5UHE0MmJqVXFseGhJa0lIbzF5UUNzNEJTQ25qQmtKamtmSHF5MWdreWY5Z0RkV3J1T0p0WVdPc2Z1am1rTQ&q=https%3A%2F%2Fpytorch.org%2Ftutorials%2Fbeginner%2Fnlp%2Fpytorch_tutorial.html&v=PaCmpygFfXo)

- PyTorch internals ref: [http://blog.ezyang.com/2019/05/pytorc...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbEI3V2g2WGg1YWVCR2Y0OGQ4SVBJUWt1MkJEZ3xBQ3Jtc0ttMXVqaWVUV3ZUSGtrOExOSjRvM0ttcnljQlMwbmZjYS1uVkRBc1Q2bkQ4ei1HaWpLVkxOSGs5Vmlaemd1eERoeEtfY0t6OEMweUFvRWZJX2FTWEVueUY3YXdYT2V0ZV9RdlZrUlFHcE5Oc3hhdzdhbw&q=http%3A%2F%2Fblog.ezyang.com%2F2019%2F05%2Fpytorch-internals%2F&v=TCH_1BHY58I)

# Tensors

A multidimensional array with many “bells and whistles”

# Broadcasting

```python
import numpy as np
A = np.array([56.0, 0.0, 4.4, 68.0],
						 [1.2, 104.0, 52.0, 8.0],
						 [1.8, 135.0, 99.0, 0.9])])
cal = A.sum(axis=0)
```

`cal` will be equal to [59, 239, 155.4, 76.9]

```python
cal = A.sum(axis=0)
percentage = 100*A/cal.reshape(1,4)
```

### Broadcasting examples:

![Untitled](The%20PyTorch%20library/Untitled.png)

Broadcasting basically just extends the smaller matrix to be the same size as the bigger one. 

![Untitled](The%20PyTorch%20library/Untitled%201.png)

### How to determine if two tensors are broadcastable

1. Align all the dimensions to the right
2. Iterate from right to left
    1. All the dimensions have to be either:
        1. Equal
        2. One of them must be one
        3. Or one of them doesn’t exist