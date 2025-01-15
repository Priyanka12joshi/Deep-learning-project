# Deep-learning-project
This project uses the popular MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9), to build a CNN model for digit classification. The model is trained on the dataset, evaluated for accuracy, and saved for future use. A sample image is also used to make predictions.
# Build the CNN Model
Step 1: Load the MNIST Dataset
The code first loads the MNIST dataset using tensorflow.keras.datasets.mnist.load_data(). The dataset is split into training and test sets:

train_images and train_labels: Training data and labels.
test_images and test_labels: Test data and labels.
 step2: Preprocess the Data
The images are reshaped into the shape (28, 28, 1) to match the input requirements for the CNN model and normalized to the range [0, 1]. Labels are converted to categorical format using tf.keras.utils.to_categorical().

Step 3: Build the CNN Model
A Convolutional Neural Network (CNN) is built using the following layers:

Conv2D Layer: 32 filters of size 3x3 with ReLU activation.
MaxPooling2D Layer: Pooling layer with a 2x2 window.
Conv2D Layer: 64 filters of size 3x3 with ReLU activation.
MaxPooling2D Layer: Pooling layer with a 2x2 window.
Conv2D Layer: 64 filters of size 3x3 with ReLU activation.
Flatten Layer: Flattens the output into a 1D vector.
Dense Layer: Fully connected layer with 64 units and ReLU activation.
Dense Output Layer: Softmax activation for a 10-class output (digits 0-9).
Step 4: Compile the Model
The model is compiled with the following parameters:

Optimizer: Adam optimizer.
Loss Function: Categorical cross-entropy (for multi-class classification).
Metrics: Accuracy metric.
Step 5: Train the Model
The model is trained on the training data for 5 epochs with a batch size of 64. The validation split is set to 10% of the training data to monitor overfitting.

