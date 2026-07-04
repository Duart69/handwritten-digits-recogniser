\# Handwritten Digits Recogniser



First approach to a neural network which recognises handwritten digits using the MNIST database, this project is built in a way to be a first contact with neural networks and deep learning, it's built following the digital free book by Michael Nielsen, if you're interested in learning deep learning or want to know how it works in depth, definitely go to check out:



http://neuralnetworksanddeeplearning.com/index.html



\---



\## 🚀 Project Overview



This project is a from-scratch implementation of a neural network capable of identifying handwritten digits (0-9) on greyscale images of 28x28 pixels. The project is made in Python using NumPy library.

This neural network was trained with the MNIST provided free database, which contains around 60,000 images of these handwritten digits. At this moment the neural network achieves an accuracy of about 94-95%.



\---



\## 🧠 Key Features

* \*\*From-Scratch Architecture:\*\* Matrix-based implementation of layers, weights and biases
* \*\*Backpropagation Algorithm:\*\* Full manual implementation of the four fundamental equations of backpropagation
* \*\*Stochastic Gradient Descent (SGD):\*\* Mini-batch based training optimization strategy
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



This would feed the neural network with the database. This run.py initialize the neural network with:

\-A hidden layer with 30 neurons

\-30 epochs

\-Mini Batch Size of 10

\-And a learning rate of 3



You will be able to see how the neural network evolves on each epochs with a ratio of x/10000 where x is the number of digits the neural networks guessed correctly out of 10,000 test images



