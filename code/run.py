import mnist_loader
import network as network

print("Loading data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)
print("Data loaded successfully!")

net = network.Network([784, 30, 10])

print("Starting training...")

net.SGD(training_data = training_data,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test_data
        )