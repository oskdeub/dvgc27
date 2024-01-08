### Machine Learning Basics

#### 1. Tensor (4 points)
**(a)** An example of a tensor representing a confusion matrix with 2 classes and 40 instances could be: 
```
[[20, 5],
 [10, 5]]
```
Here, 20 instances are correctly predicted as class 1, 5 are incorrectly predicted as class 2 when they are actually class 1, 10 are incorrectly predicted as class 1 when they are actually class 2, and 5 are correctly predicted as class 2.

**(b)** The shape of this tensor would be (2, 2) as it's a 2x2 matrix. The rank of the tensor, which indicates the number of dimensions, is 2.

#### 2. Sets (6 points)
**(a)** Training set, Validation set, Test set.

**(b)** The Training set is used to train the model, fitting the parameters to the data. The Validation set is used during the model selection phase to fine-tune hyperparameters and prevent overfitting. The Test set is used after the model has been selected and trained to assess its performance on unseen data, providing an unbiased evaluation of its generalization capability.

#### 3. Biases (2 points)
**(a)** Confirmation bias: This human bias occurs when an individual favors information that confirms their pre-existing beliefs, affecting the model if the data reflects these beliefs.
**(b)** Sampling bias: This data bias happens when the dataset is not representative of the population it's supposed to model, leading to skewed results in predictions.

### Deep Learning Practice

#### 4. General questions (fastai) (6 points)
**(a)** PyTorch tensors are used because they allow for efficient computation and automatic differentiation, which are essential for training deep learning models.
**(b)** `item_tfms = Resize(460)` resizes each item to 460 pixels on the smallest side, maintaining aspect ratio. `batch_tfms = aug_transforms(size=224, min_scale=0.5)` applies augmentation transforms to the batch, resizing images to 224 pixels and applying random scaling with a minimum scale of 0.5.

#### 5. Hyperparameters (6 points)
**(a)** Batch size influences the number of samples that will be propagated through the network at once. Optimal values depend on the available memory and problem complexity; too large may cause memory issues, too small may lead to unstable gradients.
**(b)** Learning rate influences the step size at each iteration while moving toward a minimum of the loss function. Optimal values are problem-specific; too high can overshoot the minimum, too low can cause slow convergence.
**(c)** The amount of epochs influences how many times the learning algorithm will work through the entire training dataset. Optimal values are such that further training does not improve validation metrics, avoiding overfitting.

### Deep Learning Theory

#### 6. Architecture (5 points)
**(a)** Convolution involves filtering an input with a set of learnable filters to create feature maps. Padding adds extra pixels around the input to allow for more space for the filter to cover the image. Stride is the number of pixels by which we move the filter across the input.
**(b)** Residual neural networks, or ResNets, use shortcut connections to skip one or more layers. Benefits include avoiding the vanishing gradient problem, enabling training of much deeper networks.

#### 7. General Questions (7 points)
**(a)** Generative networks are designed to generate data that is similar to some input data, often used in tasks like image generation, text-to-image synthesis, etc.
**(b)** Exploding gradients are large gradients that can cause numerical instability. They can be avoided by using gradient clipping, proper initialization, or architectures designed to mitigate them like LSTM or GRNs.
**(c)** Non-linear activation functions allow neural networks to capture complex patterns in the data. Without them, a neural network would be equivalent to a linear model.
**(d)** Step functions are not optimal because they are not differentiable, which makes it impossible to use gradient-based optimization methods.
**(e)** The use of the Sigmoid function can be problematic because it can cause vanishing gradient issues, where gradients become very small and slow down learning.
**(f)** Given an input size of 200x100x3 and a Kernel size of 3x3, each filter has 3x3x3 weights plus a bias term, so 28 parameters per filter.
**(g)** A neural network needs at least one hidden layer to fulfill the Universal Approximation Theorem, which states that a network can approximate any function given sufficient neurons.

### Classical Machine Learning

#### 8. Answer the following questions (4 points)
**(a)** Random Forest classifiers avoid overfitting by building multiple decision trees and aggregating their results, which diversifies the model and reduces variance.
**(b)** K-means clustering algorithm partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean. Unsupervised learning, such as clustering, finds structure in unlabeled data without predefined labels or outcomes.

This response aims to provide concise yet detailed answers in line with the point value and expected depth of response.