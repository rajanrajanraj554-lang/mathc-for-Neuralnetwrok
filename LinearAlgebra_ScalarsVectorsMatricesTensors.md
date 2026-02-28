# Linear Algebra: Scalars, Vectors, Matrices, and Tensors (Notations and Shapes)

## Overview
This module covers the fundamental building blocks of linear algebra used in neural networks. We'll explore the mathematical definitions, notations, shapes, and practical applications in neural networks, with examples ranging from simple to complex.

## 1. Scalars

### Mathematical Definition
A scalar is a single real number, denoted by lowercase letters $a \in \mathbb{R}$

### Notation and Shape
- Notation: Lowercase letters ($a, b, c$ or $\alpha, \beta, \gamma$)
- Shape: $()$ or $(1)$ - zero-dimensional
- Type: Single numerical value

### Simple Examples
1. $x = 5$
2. $\alpha = -2.3$
3. $learning\_rate = 0.01$

### Complex Examples
1. $loss = 0.456789$ (a loss value in neural network training)
2. $\lambda = 0.001$ (regularization parameter)
3. $\sigma = 1.0$ (standard deviation in Gaussian distribution)

### NN Applications
- Bias terms in neurons
- Learning rates in optimization algorithms
- Regularization coefficients
- Activation values of individual neurons
- Loss function outputs

## 2. Vectors

### Mathematical Definition
A vector is an ordered array of scalars, denoted by bold lowercase letters $\mathbf{x} = [x_1, x_2, ..., x_n]^T \in \mathbb{R}^n$

### Notation and Shape
- Notation: Bold lowercase letters ($\mathbf{x}, \mathbf{w}, \mathbf{b}$) or lowercase with arrow ($\vec{x}$)
- Shape: $(n,)$ where $n$ is the number of elements
- Convention: Column vectors by default (transpose indicated by $^T$)

### Simple Examples
1. $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \in \mathbb{R}^3$
2. $\mathbf{w} = \begin{bmatrix} 0.5 \\ -1.2 \\ 0.8 \end{bmatrix} \in \mathbb{R}^3$
3. $\mathbf{b} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \in \mathbb{R}^2$

### Complex Examples
1. Input vector to a neural network: $\mathbf{x} = \begin{bmatrix} pixel_1 \\ pixel_2 \\ \vdots \\ pixel_{784} \end{bmatrix} \in \mathbb{R}^{784}$ (for 28×28 image flattened)
2. Weight vector in a single neuron: $\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix} \in \mathbb{R}^d$
3. Activation vector of a hidden layer: $\mathbf{h} = \begin{bmatrix} h_1^{(1)} \\ h_2^{(1)} \\ \vdots \\ h_n^{(1)} \end{bmatrix} \in \mathbb{R}^n$

### NN Applications
- Input data representations (flattened images, feature vectors)
- Weight vectors for individual neurons
- Bias vectors for layers
- Activation vectors representing outputs of entire layers
- Embedding vectors in NLP models

## 3. Matrices

### Mathematical Definition
A matrix is a 2D array of scalars, denoted by bold uppercase letters $\mathbf{A} \in \mathbb{R}^{m \times n}$

### Notation and Shape
- Notation: Bold uppercase letters ($\mathbf{A}, \mathbf{W}, \mathbf{X}$)
- Shape: $(m, n)$ where $m$ is rows and $n$ is columns
- Element notation: $A_{ij}$ refers to element in $i$-th row, $j$-th column

### Simple Examples
1. $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$
2. $\mathbf{B} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$
3. $\mathbf{C} = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix} \in \mathbb{R}^{3 \times 1}$ (column matrix/vector)

### Complex Examples
1. Weight matrix for fully connected layer: $\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \cdots & w_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$
2. Batch of input samples: $\mathbf{X} = \begin{bmatrix} \text{-- } \mathbf{x}_1^T \text{ --} \\ \text{-- } \mathbf{x}_2^T \text{ --} \\ \vdots \\ \text{-- } \mathbf{x}_b^T \text{ --} \end{bmatrix} \in \mathbb{R}^{b \times d}$ where $b$ is batch size
3. Covariance matrix: $\mathbf{\Sigma} = \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1d} \\ \sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{d1} & \sigma_{d2} & \cdots & \sigma_d^2 \end{bmatrix} \in \mathbb{R}^{d \times d}$

### NN Applications
- Weight matrices connecting layers
- Batch input data representation
- Jacobian matrices in backpropagation
- Attention matrices in transformer models
- Covariance matrices in batch normalization

## 4. Tensors

### Mathematical Definition
A tensor is a multi-dimensional array, generalizing scalars, vectors, and matrices to arbitrary dimensions. A tensor of order $k$ has $k$ indices.

### Notation and Shape
- Notation: Bold uppercase letters ($\mathbf{T}, \mathbf{X}, \mathbf{W}$) or with specific indexing
- Shape: $(d_1, d_2, ..., d_k)$ where $k$ is the number of dimensions
- Element notation: $T_{i_1, i_2, ..., i_k}$ refers to element with indices $i_1, i_2, ..., i_k$

### Simple Examples
1. 3D tensor: $\mathbf{T} \in \mathbb{R}^{2 \times 3 \times 4}$ with elements $T_{ijk}$ where $i \in [1,2]$, $j \in [1,3]$, $k \in [1,4]$
2. Rank-4 tensor: $\mathbf{X} \in \mathbb{R}^{10 \times 3 \times 28 \times 28}$ (batch of 10 RGB images of size 28×28)

### Complex Examples
1. Convolutional layer weights: $\mathbf{W} \in \mathbb{R}^{c_{out} \times c_{in} \times k_h \times k_w}$
   - $c_{out}$: number of output channels
   - $c_{in}$: number of input channels  
   - $k_h, k_w$: kernel height and width
   
2. Batch of video frames: $\mathbf{V} \in \mathbb{R}^{b \times t \times c \times h \times w}$
   - $b$: batch size
   - $t$: number of time steps (frames)
   - $c$: color channels
   - $h, w$: height and width of frames
   
3. Attention tensor in transformers: $\mathbf{A} \in \mathbb{R}^{b \times n \times h \times s \times s}$
   - $b$: batch size
   - $n$: number of attention heads
   - $h$: head dimension
   - $s$: sequence length

### NN Applications
- Image data (RGB channels): $\mathbb{R}^{batch \times channels \times height \times width}$
- Video data: $\mathbb{R}^{batch \times time \times channels \times height \times width}$
- Convolutional filter weights
- Multi-head attention weights in transformers
- Hidden states in RNNs across time steps

## Dimension Relationships in Neural Networks

### Common Shape Patterns

#### Fully Connected Layer
Input: $\mathbf{x} \in \mathbb{R}^{n}$, Weight: $\mathbf{W} \in \mathbb{R}^{m \times n}$, Output: $\mathbf{y} \in \mathbb{R}^{m}$
Computation: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$

#### Batch Processing
Input: $\mathbf{X} \in \mathbb{R}^{b \times n}$, Weight: $\mathbf{W} \in \mathbb{R}^{m \times n}$, Output: $\mathbf{Y} \in \mathbb{R}^{b \times m}$
Computation: $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{B}$ (where $\mathbf{B}$ is bias broadcasted)

#### Convolutional Layer
Input: $\mathbf{X} \in \mathbb{R}^{b \times c_{in} \times h_{in} \times w_{in}}$
Weight: $\mathbf{W} \in \mathbb{R}^{c_{out} \times c_{in} \times k_h \times k_w}$
Output: $\mathbf{Y} \in \mathbb{R}^{b \times c_{out} \times h_{out} \times w_{out}}$

## Key Takeaways

1. **Scalars** are single values used for learning rates, biases, and individual measurements
2. **Vectors** represent ordered collections like feature vectors or layer activations
3. **Matrices** handle transformations between layers and batched data
4. **Tensors** generalize to higher dimensions for complex data like images and videos
5. Understanding shapes is crucial for proper dimension matching in neural network operations
6. The notation and shape conventions ensure consistency in mathematical expressions and implementations

## Practical Tips for Neural Network Implementation

1. Always verify dimensional compatibility in operations (e.g., matrix multiplication requires inner dimensions to match)
2. Remember that frameworks often use different default orientations (row vs. column vectors)
3. Batch processing typically adds an extra leading dimension
4. Keep track of which dimensions correspond to batch size, features, channels, etc.