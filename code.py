import numpy as np


def sigmoid_activation(x, takeDerivate = False):
    
    if(takeDerivate == True):
        return sigmoid_activation(x) * (1 - sigmoid_activation(x))
    
    return 1 / (1 + np.exp(-x))



def print_params(iter, x_train, expected_labels, weights, prediction):
    print("-----------------------------------------------------------\n")
    print("iteration # ", iter)
    print("Input data is... \n", x_train)
    print("Expected labels are... \n", expected_labels)
    print("Current weights are... \n", weights)
    print("Predictions for train data at this iteration are... \n", prediction)
    print("-----------------------------------------------------------\n")
#https://github.com/Lodur03/Perceptron/blob/master/perceptron.py
#    https://github.com/acharles7/Perceptron-/blob/master/ml.ipynb
def trainPerceptron(inputs, t, weights, rho, iterNo):
    
    for iter in range(iterNo):
        for i in range(inputs.shape[0]):
#            Calculate Output
            output = np.dot(inputs[i],weights)
#            Calculate sigmoid activation value
            sigmoid_value = sigmoid_activation(output)
#            Calculate Error
            error = (t[i]-sigmoid_value)
#            Caclculate refers to derivative of sigmmoid function
            sigmoidDerivative = sigmoid_activation(sigmoid_value,True)
#            update all weights with new weights
            weights[:1] += rho * error * sigmoidDerivative * inputs[i,-1]
#            if(iter % 100 == 0):
#                print_params(iter, inputs[i], t[i], weights, sigmoid_value)
    print(weights)
    return weights

def testPerceptron(sample_test, weights):

    y = 0   
    #write prediction code in here
    #
    #end your code
    weights = np.reshape(weights,3073)
    result = np.dot(weights,sample_test)
    acc = sigmoid_activation(result)
    print ("result : ",result)
    print("acc : ",acc)
    
  
    return y
 
#######our main code
np.random.seed(1)

from keras.datasets import cifar10    
(x_data, y_train), (x_test, y_test) = cifar10.load_data()


x_data = x_data.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

idx1 = np.array(np.where(y_train==0)).T
idx2 = np.array(np.where(y_train==1)).T
n1 = idx1.shape[0]
n2 = idx2.shape[0]
n = n1+n2
t = np.zeros((n,1), np.int32)
t[0:5000] = y_train[idx1[:,0]]
t[5000:10000] = y_train[idx2[:,0]]

x_train = np.zeros((n,3072), np.uint8)
x_train[0:5000,:] = x_data[idx1[:,0],:]
x_train[5000:10000,:] = x_data[idx2[:,0],:]

bias_x = np.ones((10000,1), np.uint8)
x_train = np.append(x_train, bias_x, axis=1)

weights = 2*np.random.random((3072,1)) - 1

bias = np.ones((1,1), np.uint8)
weights = np.append(weights, bias, axis=0)

iterNo=100
rho = 0.0001
sample_test = x_test[3,:]
sample_test = np.append(sample_test,1)
sample_test = np.reshape(sample_test,3073)

weights = trainPerceptron(x_train, t, weights,rho, iterNo)


expected = y_test[3]
predicted = testPerceptron(sample_test, weights)

