\# Handwritten Digits Recogniser



First approach to a neural network which recognises handwritten digits using the MNIST database, this project is built in a way to be a first contact with neural networks and deep learning, it's built following the digital free book by Michael Nielsen, if you're interested in learning deep learning or want to know how it works in depth, definitely go to check out:



http://neuralnetworksanddeeplearning.com/index.html



\---



\## 🚀 Project Overview



This project is a from-scratch implementation of a neural network capable of identifying handwritten digits (0-9) on greyscale images of 28x28 pixels. The project is made in Python using NumPy library.

This neural network was trained with the MNIST provided free database, which contains around 60,000 images of these handwritten digits. The repository features two distinct versions of the network architecture to demonstrate performance progression:

0: The simpler implementation. It uses no advanced optimization techniques and relies on a Quadratic Cost Function and standard normal weight initialization.

1: An improved version that implements a Cross Entropy Cost Function, regularization (Weight Decay) to minimize overfitting, and normalized weight initialization to prevent neuron saturation.


\---



\## 🧠 Key Features

* \*\*From-Scratch Architecture:\*\* Matrix-based implementation of layers, weights and biases
* \*\*Backpropagation Algorithm:\*\* Full manual implementation of the four fundamental equations of backpropagation
* \*\*Stochastic Gradient Descent (SGD):\*\* Mini-batch based training optimization strategy
* \*\*Performance Improvements:\*\* Side-by-side comparison showcasing faster learning and better generalization metrics.
* \*\*Mathematical Stability:\*\* Features robust activation functions optimized to handle potential numerical overflows.



\---



\## 🛠️ Installation \& Setup



1.\*\*Clone the repository\*\*

```bash

git clone https://github.com/Duart69/handwritten-digits-recogniser.git

cd handwritten-digits-recognizer

```



2.\*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



3.\*\*Run the training script\*\*

```bash

python code/run.py

```



When you run run.py, the CLI will prompt you to choose between the two network versions (0 or 1). Once chosen, the script feeds the data into the network initialized with the following hyperparameter defaults:

\-Topology: 1 Hidden layer with 30 neurons (784 -> 30 -> 10)

\-Training Length: 30 epochs

\-Mini Batch Size: 10

\-Learning rate: 0.2

\-Regularization Parameter: 0.5 (Only active if Network 1 is selected)



The terminal will print live evaluations after each epoch, displaying the total cross-entropy or quadratic cost alongside a real-time accuracy ratio (e.g., x / 10000, where x represents the number of images correctly predicted out of the 10,000-image test set).



