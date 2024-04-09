---
title: PyTorch Tensors
author: Kamal Acharya
pubDatetime: 2024-04-09T04:02:40.479Z
slug: pytorch-tensors
featured: true
draft: false
tags:
  - AI
  - PyTorch
ogImage: ""
description: Basic fundamental operations with PyTorch tensors
---

PyTorch Tensors here we come

## Table of Contents

## Disclaimer:

This resource has been curated after I completed the [Learn PyTorch Course](https://www.learnpytorch.io)  
All of the materials and code snippets are based on the course.

**The credit for most of the content here goes to the creators of [Learn PyTorch Course](https://www.learnpytorch.io).**

My work here is an extension to their material and adding on to concepts which I don't fully understand.
Also, this article (more to come) serves to me as my own cheatsheet when I am working on PyTorch projects.

## Step 0 - Imports

Basics:

```python
import torch
torch.__version__
```

## Motivation behind Tensors

You can think about tensors as normal python arrays that can be used to represent matrixes and ultimately perform matrix multiplication.

**Why matrices?**
Essentially, a neural network can only compute numbers.
And boiling it down even more, it's usually lots of matrix multiplication and matrix manipulation.

All of the data inputs that you feed into a neural network are numbers - be it an image, an audio or text; computers can only compute numbers.
Example:

- You can represent an image with a tensor: `[3, 224, 224]` for `[color_channel, width, height]`

Since the essence of a neural network is just matrix multiplication and it's manipulation, you **tensors** help us with things that deal with matrix.

Plus, when you put data into a tensor, it is easier for the GPU to do it's computation. Therefore, it is essential that every ML/AI engineer be comfortable with working with tensors.

**Coming soon:**

- [ ] How a neural network learns (matrix multiplication) --> feed forward & back-propagation

## Basic Tensor Operation

### scalar, vector, matrix/tensor, dims & shape

Concepts to understand:

- dimensions of a tensor
- Shape of a tensor
- scalar
- vector
- Matrix

```python
# Creating a tensor
scalar = torch.tensor(7)
print(scalar) # Output: tensor(7)

print(scalar.ndim) # Output: 0

# Retrieve number: item()
# Only works with one element tensor
print(scalar.item()) # Output: 7

# VECTOR =======
vector = torch.tensor([7, 7])
print(vector) # Output: tensor([7 , 7])

print(vector.ndim) # Output: 1

# MATRIX =======
MATRIX = torch.tensor([[7, 8],
							  [9, 10]])
print(MATRIX) # Output: tensor([[7, 8],
				  #				      9, 10])
print(MATRIX.ndim) # Output: 2
print(MATRIX.shape) # Output: torch.Size(2, 2)

# A tensor example:
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

print(TENSOR.ndim) # 3
print(TENSOR.shape) # torch.Size([1, 3, 3])
```

> **LifeHacks:**
> You can tell the number of dimensions a tensor in PyTorch has by the number of square brackets on the outside (`[`) and you only need to count one side.

> Note:
> you'll often see scalars and vectors denoted as lowercase letters such as `y` or `a`. And matrices and tensors denoted as uppercase letters such as `X` or `W`.

### Random Tensors, Zeros, Ones, Ranges

Why do you need tensors with random numbers?

In essence the process to train a neural network is:
`Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...`

- Therefore you need to know different ways in which you can work with tensors

Getting info from tensor: (the most common attributes you will want to know)

- `shape`: What is the shape of the tensor?
- `dtype`: What is the datatype of the tensor?
- `device`: Where is the tensor located - CPU or GPU?

```python
# use torch.rand() & size parameter
random_tensor = torch.rand(size=(3,4))
print(random_tensor, random_tensor.dtype)
# Output: tensor[[0.6, ...]], torch.float32

# Tensors filled with zeros or ones
zeros = torch.zeros(size=(3,4))
print(zeros.dtype) # Output: float32

ones = torch.ones(size=(3,4))
print(ones.dtype) # Output: torch.float32

# Range of tensors:
# torch.arange(start, end, step)
# NOTE: torch.range() is deprecated
zero_to_ten = torch.arange(start=0, end=10, step=1)
# [0,1,2...,9]

# One tensor of certain type with the same shape as another tensor
# E.g: a tensor of all zeros with the same shape as a previous tensor
# torch.zeros_like(input_tensor)
# OR
# torch.ones_like(input_tensor)
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros) # Output: tensor([0,0,0,0,0,0,0,0,0,0])
```

Helper function to display the tensor info

```python
def print_tensor_info(i_tensor):
	print('=======================')
	print('Tensor information')
	print(i_tensor)
	print(f'Shape: {i_tensor.shape}')
	print(f'dtype: {i_tensor.dtype}')
	print(f'Device: {i_tensor.device}')
	print('=======================')
```

### Manipulating Tensors

**Fundamental operations:**

```python
# addition, subtraction & multiplication
tensor = torch.tensor([1, 2, 3])
tensor + 10 # tensor([11, 12, 13])
tensor * 10 # tensor([10, 20, 30])
tensor - 10 # tensor([-9, -8, -7])
print(tensor) # tensor([1, 2, 3]) --> Doesn't change unless reassigned

# Built-in functions
# torch.mul(), torch.add()
torch.multiply(tensor, 10) # same as torch.mul(tensor, 10)
torch.add(tensor, 10)

# NOTE: more common to use * instead of torch.mul()

# Element-wise multiplication
tensor * tensor # tensor([1,4,9])
```

**Coming Soon:**

- [ ] Visualizing Matrix multiplication

**Matrix Multiplication:**

- One of the most common operation in ML/AI is matrix multiplication
- PyTorch implements it using `torch.matmul()`

**Rules:**

- inner dimensions must match `(2,3) @ (3,2)`
- The resulting matrix has the shape of outer dimension: `(2,3) @ (3,2) --> (2,2)`

```python
tensor = torch.tensor([1,2,3])
tensor.shape # torch.Size([3])

# Element-wise
tensor * tensor # tensor([1,4,9])
# Matrix multiplication
torch.matmul(tensor, tensor) # tensor(14)
# NOTE: above is same as: torch.mm()
```

**Torch Transpose**

```python
# Shape error example:
t_A = torch.tensor([[1,2],
						 [3,4],
						 [5,6]], dtype=torch.float32)

t_B = torch.tensor([[1,2],
						 [3,4],
						 [5,6]], dtype=torch.float32)

torch.matmul(t_A, t_B) # Error
# (3, 2) @ (3, 2): Inner dimension won't work. Need to transpose one

# torch.transpose(input, dim0, dim1)
# input: tensor,
# dim0 and dim1: dims to be swapped
# ===== OR =====
# input_tensor.T:

torch.matmul(t_A.T, t_B) # No error
```

**Aggregation: min, max, mean, sum**

```python
x = torch.arange(0, 100, 10)
# Aggregation
x.min() # 0
x.max() # 90
x.mean() # 45.5 # might require float32 dtype
x.sum() # 450

# Can do the same with torch methods
torch.max(x)
torch.min(x)
torch.mean(x.type(torch.float32)) # converting dtype
torch.sum(x)

# Positional min/max
x.argmax() # index where maximum occurs
x.argmin() # index where minimum occurs
```

**Changing Datatypes:**

```python
# Syntax: torch.Tensor.type(dtype=None)
# Default = float32
t = torch.arange(10.,100.,10.)
t.dtype # torch.float32

# Convert and assign to t1
t1 = t.type(torch.float16)
t1.dtype # torch.float16
```

### Reshaping, Stacking, Squeezing & Unsqueezing

**Why:**
You will have situations where you want to reshape/change dimensions of a tensor _**without changing it's values**_

```python
# torch.reshape(input, shape) # torch.Tensor.reshape()
# - reshape input to shape if compatible

# Tensor.view(shape)
# - Return a view of original tensor in a different shape (retains same value)

# torch.stack(tensors, dim=0)
# - Concatenate sequences of tensors along a new dimension?
# - all tensors must be same
# NOTE: Need visualization: Don't understand

# torch.squeeze(input)
# - removes all dimensions that have value 1
# NOTE: Need visualization: Don't understand

# torch.unsqueeze(input, dim)
# - return input with dim=1 added at dim
# NOTE: Need visualization: Don't understand

# torch.permute(input, dims)
# - return a view of original input with dims rearranged to dims)
# NOTE: Need visualization: Don't understand
```

**TODO: Add visualizations**
**TODO: Research more on the methods**

### Indexing

- Indexing on tensors should be similar to python lists or NumPy arrays

```python
x = torch.arange(1, 10).reshape(1,3,3)
print(x)
# Output:
# (tensor([[[1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9]]]),
print(x.shape)
# Output:
# torch.Size([1, 3, 3]))

x[0] # first sq bracket
x[0][0] # second sq bracket [1,2,3]
x[0][0][0] # third sq bracket 1

# Can use : to specify all
x[:, 0] # get all vals of 0th dim & 0 index of 1st dim
# Output: tensor([[1,2,3]])

x[:, :, 1] # Get all vals of 0th & 1st dim and only 1th index
# Output: tensor([[2,5,8]])

x[:, 1, 1] # Get all vals of 0 dim but only 1 index of 1st and 2nd dim
# Output: tensor([5])

# Get index 0 of 0th and 1st dim and all vals of 2nd dim
x[0, 0, :] # Same as x[0][0]
```

**TODO: Visualize Indexing**

### PyTorch tensor & NumPy

**Conversions:**

- `torch.from_numpy(ndarray)` --> array to tensor
- `torch.Tensor.numpy()` --> tensor to array

> **Note:** By default NumPy arrays are created with `float64`
> If you convert it to tensor, the `dtype` is maintained.

```python
# Array to tensor
a = np.arange(1.0, 8.0)
t = torch.from_numpy(a) # dtype remains float64

# Tensor to Array
t = torch.ones(7)
a = t.numpy() # float32 is maintained unless changed
```

**Reproducability:**

```python
import rand
import torch

RANDOM_SEED=10
torch.manual_seed(seed=RANDOM_SEED)
```

## GPU!

`!nvidia-smi` to check if a gpu has been properly setup

```python
# Check for gpu
import torch
torch.cuda.is_available()

# set device type:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# devices count
torch.cuda.device_count()

# Check where the tensors are
tensor = torch.ones(2)
print(tensor.device) # Default is 'cpu'

# Move to GPU
t_on_gpu = tensor.to('gpu') # or device variable

# NOTE: Cannot numpy() on tensors in a GPU
t_on_gpu.numpy() # error
# Send gpu tensor on cpu
t_on_cpu = t_on_gpu.cpu().numpy() # No error
```
