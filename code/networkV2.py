import numpy as np
import random

def sigmoid(z):
    return (1.0/(1.0 + np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cross_entropy_cost(a,y):
    return -np.sum(np.nan_to_num((y*np.log(a)) + ((1-y)*np.log(1-a))))

class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.numLayers = len(sizes)
        self.biases = []
        for i in sizes[1:]:
            self.biases.append(np.random.randn(i,1))
        
        self.weights =  [(np.random.randn(y,x))/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size,
            eta, test_data = None, reg_param = 0.0):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):

            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch,eta,reg_param,n)

            if test_data:
                evaluate, cost = self.evaluate(test_data, reg_param)
                print("Epoch {0}: {1} / {2}\nTraining Cost:{3:.2f}".format(
                    i, evaluate, n_test, cost))
            else:
                print("Epoch {0} complete\n".format(i))
                
    
    def updateMiniBatch(self, mini_batch, eta, reg_param, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-(reg_param*eta/n))*w - (eta/len(mini_batch)*nw)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta/len(mini_batch)*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
        
    def evaluate(self, test_data, reg_param):
        total_cost = 0.0
        test_result = []
        for x, y in test_data:
            output = self.feedforward(x)
            test_result.append((np.argmax(output),y))

            if np.isscalar(y):
                y_vector = np.zeros((10,1))
                y_vector[y] = 1.0
            else:
                y_vector = y

            total_cost += cross_entropy_cost(output,y_vector)
        average_cost = total_cost/len(test_data)
        weight_square_sum = sum(np.sum(w**2) for w in self.weights)
        regularization_cost = (reg_param/(2*len(test_data)))*weight_square_sum
        return (sum(int(x==y) for x, y in test_result), average_cost + regularization_cost)
