# handwrittenDigitClassifier
Forward Propagation:
1. I am using a database that has already convirted hundreds of 28x28 pixel images of handwritten numbers into a  one dimensional matrix of 784 pixels overall. Each of these pixels can have values ranging from 0-255, 0 representing an empty pixel and 255 representing a completley colored and dark pixel.  
2. I then transposed this one dimensional matrix to become a matrix with 784 row elements and one column element for each. This was done to treat each specific pixel as an individual node that has the ability to hold a weight and a bias. This is one of the fundamental components of a neural network and will help us with our predictions. 
3. The first hidden layer is be a 10x1 matrix representing each of the 10 digits the images can be showing. One hidden layer is enough for a project like this since it lacks excessive complexity. 
4. I then used another 10x1 matrix as the output layer to get the result of the evaluations computed by the neural network. 
5. The activation function I used was as ReLu (return x if x > 0 else 0). This function, as well as other such as tanh, enamble our model to behave non-linearly. In turn, our model can now find ways to better fit real world data, which presents non-linear relationships, enhancing the models ability to represent complex patterns. 
6. The output layer uses a simple softmax activation function that takes as inputs the previous activated layer. This layer displayes in the form of probabilities, the predictions of the handwritten numbers. 

Backwards Propagation:

