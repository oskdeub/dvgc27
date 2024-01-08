Certainly, I will provide answers according to the point values assigned to each question in the examination paper for the course "Applied Machine Learning – DVGC27."

### Machine Learning Basics

#### 1. Sets
**(a) How, when, and why is the test set used?** (2 points)
- The test set is used after the machine learning model has been trained and validated. It is utilized to evaluate the final model's performance, reflecting how well the model will generalize to new, unseen data. The test set is crucial for providing an unbiased assessment of a final model fit.

**(b) How, when, and why is the validation set used?** (2 points)
- The validation set is used during the model training process to fine-tune model parameters and to provide an unbiased evaluation of a model fit. It's utilized after the model is trained on the training set and before the final evaluation on the test set. This helps in selecting the best model and avoiding overfitting.

#### 2. Metrics
**(a) What are precision and accuracy? What are the differences between them?** (2 points)
- Precision measures the ratio of true positives to the total predicted positives. Accuracy measures the ratio of correctly predicted observations to the total observations. Precision is used when the cost of a false positive is high, while accuracy provides a general measure of model performance.

**(b) Draw a confusion matrix with 3 classes and an accuracy of 75%.** (2 points)
- A 3x3 confusion matrix for three classes, where the sum of the diagonal elements (true positives for each class) divided by the total number of instances equals 75%, would illustrate this. For example, if there are 100 instances in total, 75 would be correctly classified.

#### 3. General
**(a) What are the differences between unsupervised and supervised learning?** (2 points)
- Supervised learning uses labeled data to train models, aiming to predict output variables from input variables. Unsupervised learning works with unlabeled data to find structure, such as clusters or associations.

**(b) What is overfitting and how do we avoid it?** (2 points)
- Overfitting occurs when a model learns the training data too well, including noise and fluctuations, and performs poorly on unseen data. It can be avoided by techniques such as cross-validation, regularization, and pruning (in decision trees), among others.

### Deep Learning Practice

#### 4. General questions (fastai) (6 points)
**(a) What happens if we use a batch size of 1?**
- Using a batch size of 1, known as stochastic gradient descent, means the model updates weights after every sample. This can lead to high variance in the updates, potentially causing a loss in generalization and an increase in training time.

**(b) What is the difference between resnet18 and resnet34?**
- The difference between ResNet18 and ResNet34 is the number of layers; ResNet18 has 18 layers, and ResNet34 has 34. More layers can potentially capture more complex patterns, but also require more data and computational power, and may risk overfitting.

**(c) What are U-nets and when are they used?**
- U-Nets are a type of convolutional network designed for biomedical image segmentation. They have a U-shaped architecture that enables precise localization using a combination of down-sampling and up-sampling paths.

**(d) What are the differences between item_tfms and batch_tfms and give an example of how they are used.**
- `item_tfms` are applied to each item individually before batching, while `batch_tfms` are applied to a batch of items. For example, `item_tfms` might resize images, while `batch_tfms` could normalize the entire batch.

#### 5. Mechanisms (6 points)
**(a) Describe transfer learning, how it is used, and why we use it.**
- Transfer learning involves taking a pre-trained model and fine-tuning it on a new, often smaller dataset. It is used to leverage learned features from large datasets, thus reducing the need for extensive computational resources and large labeled datasets.

**(b) Describe augmentation, how it is used, and why we use it.**
- Augmentation artificially increases the diversity of the dataset by applying random transformations, like rotation and flipping, to the training data. This helps the model generalize better by simulating a variety of scenarios.

### Deep Learning Theory

#### 6. Architecture (4 points)
**(a) What are the advantages and disadvantages of having more layers in a neural network?**
- Additional layers in a neural network allow it to capture more complex features and relationships in the data. However, they also increase the risk of overfitting, require more data to train effectively, and increase computational cost.

**(b) Why do we use convolution layers and shortcut layers (e.g., in ResNets)?**
- Convolution layers are used to capture local patterns through shared weights in an image, such as edges and textures, efficiently. Shortcut layers, or skip connections, in ResNets help mitigate the vanishing gradient problem by allowing gradients to flow through an alternative shorter path during backpropagation, enabling the training of deeper networks.

#### 7. General questions (8 points)
**(a) Why do we need nonlinear activation functions?**
- Nonlinear activation functions allow neural networks to capture complex patterns in the data. Without them, a neural network, regardless of its depth, would behave just like a single linear model.

**(b) When changing the filter-size in convolution layers, how does that change the amount of trainable parameters?**
- Increasing the filter size increases the number of trainable parameters as there are more weights per filter. Conversely, reducing the filter size reduces the number of parameters, which can reduce computational load and the chance of overfitting.

**(c) What happens if the learning rate is too high?**
- If the learning rate is too high, the model might overshoot the minimum of the loss function, causing the training to diverge and potentially never converge to an optimal solution.

**(d) What happens if the learning rate is too low?**
- If the learning rate is too low, training can become very slow and the model may get stuck in local minima, potentially leading to suboptimal performance.

**(e) Which parts of the Backpropagation Algorithm are used for inference?**
- None; backpropagation is only used for training to calculate the gradient of the loss function with respect to each weight. Inference only involves the forward pass.

**(f) What would happen if we would use Accuracy as a loss function? Why is that?**
- Accuracy is not a differentiable function, which means its gradient cannot be computed, and hence it cannot be optimized using gradient-based methods. It would make the training process impossible.

### Classical Machine Learning

#### 8. Answer the following questions (4 points)
**(a) What happens if a tree classifier is allowed to grow too deep? Describe what happens and what are the likely problems.**
- Allowing a tree classifier to grow too deep can lead to overfitting where the model captures noise in the training data. This typically results in poor generalization to new data and a complex model that is hard to interpret.

**(b) Describe Bagging in the Random Forest learner. What are the benefits?**
- Bagging, or Bootstrap Aggregating, involves training

 multiple trees on different subsets of the data and then averaging the predictions. This improves model robustness, reduces variance, and avoids overfitting.

This should provide an overview of the expected answers in line with the point values assigned.