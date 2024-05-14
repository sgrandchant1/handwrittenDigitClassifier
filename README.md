# Handwritten Digit Classifier

This project develops a neural network-based classifier that identifies handwritten digits using images formatted as 28x28 pixels. The model utilizes a simple yet effective architecture suitable for understanding basic principles of machine learning. This project purposfully AVOIDS the use of libraries such as Pytorch and Tensorflow in an attempt to deeply understand the fundamental concepts of machine learning. This project used as refference GeeksForGeeks and a Youtube video by Samson Zhang which was extremley usefull. Here are the reffrerences:

1. https://www.geeksforgeeks.org/the-role-of-weights-and-bias-in-neural-networks/
2. https://youtu.be/w8yWXqWQYmU?si=U03g9kUF5sDqMFJX

This README provides an overview of the model's structure, setup instructions, and how to use the project for classification tasks.

## Technologies Used

- Python 3.10.9
- NumPy

# Description

## Forward Propagation:

1. I am using a database that has converted hundreds of 28x28 pixel images of handwritten numbers into a one-dimensional matrix of 784 pixels overall. Each pixel can have values ranging from 0 to 255, where 0 represents an empty pixel and 255 represents a completely colored and dark pixel.
2. I then transposed this one-dimensional matrix to become a matrix with 784 row elements and one column element for each. This transformation was done to treat each specific pixel as an individual node capable of holding a weight and a bias, which are fundamental components of a neural network and aid in our predictions.
3. The first hidden layer is a 10x1 matrix representing each of the 10 digits the images could be showing. One hidden layer is sufficient for a project of this scale, which lacks excessive complexity.
4. Another 10x1 matrix is used as the output layer to obtain the results of the evaluations computed by the neural network.
5. The activation function I used was ReLU (return x if x > 0 else 0). This function, along with others such as tanh, enables our model to behave non-linearly. Consequently, our model can better fit real-world data, which often presents non-linear relationships, thus enhancing the model's ability to represent complex patterns.
6. The output layer uses a softmax activation function that takes as inputs the previously activated layer. This layer displays, in the form of probabilities, the predictions of the handwritten numbers.

## Backward Propagation:

1. The loss of the function is computed at the end of the forward propagation and once a result is outputted. The loss is simply calculated by obtaining the difference between the expected output and the obtained output.
2. I then calculated how much the nodes had contributed to the error and adjusted the weights to better fit the desired output using the calculated gradients. This is done by computing the partial derivative of the error with respect to each node. The gradient then informs us of the direction and magnitude of change needed to mitigate the loss at the current iteration.
3. Similarly, biases are also updated using the gradient. Each node is adjusted to better fit the desired result. We then continue to propagate the adjustments backward to reach the earlier layers.
4. We make sure to update our parameters for the next iteration.
