import mnist_loader
import network as network
import networkV2 as networkV2

correct_operation = False
print("\n\n-----------CHOOSE A NETWORK-----------\n\n0: This Network uses the Quadratic Difference Cost Function\n\n1:An improved version of the previous network. This uses a Cross-Entropy Cost Function, implements REGULARIZATION and sets up better initial weight initialization\n")

while not correct_operation:
        try:
                chosen = int(input("Enter your choice (0 or 1): "))
                if chosen == 0 or chosen == 1:
                        correct_operation = True   
                else:
                        raise ValueError()
        except ValueError:
                print("\nERROR: INVALID OPERATION")
print("\nLoading data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)
print("Data loaded successfully!")
print("Starting training...")
if(chosen==0):
        net = network.Network([784, 30, 10])
        net.SGD(training_data = training_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        test_data=test_data
        )
else:
        net = networkV2.Network([784,30, 30, 10])
        net.SGD(training_data = training_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.2,
        test_data=test_data,
        reg_param = 5
        )