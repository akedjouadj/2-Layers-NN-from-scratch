import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def initialisation(n0, n_neurons, n2):
    """
    function to initialise the perceptron:
    
    Arguments : 
    X : the data, numpy array

    return :
    (W,b): a tuple, the model's parameters, W is the initialized weights vector and b is the initialized bias  
    """
    W1=np.random.randn(n_neurons, n0) 
    b1=np.random.randn(n_neurons, 1)
    W2=np.random.randn(n2, n_neurons) 
    b2=np.random.randn(n2, 1)

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters

def forward_propagation(X, parameters):
    """
    function to run the model, compute the operation XW+b
    
    Arguments : 
    X : the data, numpy array
    W : the initialized weights vector, numpy array of 1 column
    b : the initialized bias, scalar

    return :
    A: the vector of probabilities for each data (each row of X) to belong to the class 0 or the class 1  
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #print(W1.shape)
    #print(X.shape)
    Z1=W1@X+ b1
    A1=1/(1+np.exp(-Z1))
    Z2=W2.dot(A1)+b2
    A2=1/(1+np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations

def log_loss(A, y, epsilon=1e-15):
    """
    function to compute the model loss
    
    Arguments : 
    A : the model output
    y : the real target vector, numpy array of 1 column, with values in {0, 1}
    epsilon : a small positive constant to ensure that you compute the logarithm of a non-null quantity
    
    return : The bernoulli log-likelihood loss between A and y  
    """

    #print(A.shape)
    #print(y.shape)
    return (1/y.shape[1])*np.sum( -y*np.log(A+epsilon) -(1-y)*np.log(1-A+epsilon) )

def back_propagation(X, y, activations, parameters):
    """
    function to compute the model gradients
    
    Arguments : 
    A : the model output
    X : the data, numpy array
    y : the real target vector, numpy array of 1 column, with values in {0, 1}

    return :
    (dW,db): a tuple, the log_loss gradients  
    """
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parameters['W2']

    m = y.shape[1]

    dZ2 = A2-y
    dW2 = (1/m)* dZ2.dot(A1.T)
    db2 = (1/m)* np.sum(dZ2, axis = 1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2)* A1 * (1-A1)
    dW1 = (1/m)* dZ1.dot(X.T)
    db1 = (1/m)* np.sum(dZ1, axis = 1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

def update(gradients, parameters, learning_rate):
    """
    function to update the model's parmeters W and b with gradient descent algorithm
    
    Arguments :
    dW, db : the gradients output
    W, b : the model parameters
    learning_rate : the model learning rate

    return :
    (W,b): a tuple, the updated model's parameters  
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1=W1- learning_rate * dW1
    b1=b1- learning_rate * db1
    W2=W2- learning_rate * dW2
    b2=b2- learning_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters

def predict(X, parameters, proba = False):
    """
    function to make a prediction with the model
    
    Arguments : 
    X : the data for prediction, numpy array
    W, b : the trained model's parameters
    proba : boolean, if True, the probability will be return instead of the class 

    return : a vector of the prediction, with values in {0,1}
    """
    activations =forward_propagation(X, parameters)
    A2 = activations['A2']
    if proba==False:
        return (A2>=0.5)*1
    else:
        return A2   


def accuracy(y, y_pred):
    """
    function to evaluate the prediction's accuracy

    Arguments : 
    y : the true target
    y_pred : the predicted target

    return : the accuracy score of the prediction, scalar between 0 and 1
    """
    return np.sum((y==y_pred)*1)/y.shape[1]

def frobenius_norm_list(l):
    """
    Compute the sum of the frobenius norms of all matrix in l   

    Arguments : 
    l : list of matrix

    return : a scalar, the sum
    """

    s = 0
    for i in l:
        s+= np.sqrt(np.sum(i**2))
    
    return s

def NN_2_layers(X_train, y_train, X_val, y_val, n_neurons = 2, learning_rate=0.001, err=1e-5, max_iter=10000, displaying_step = 1000):
    """
    function to train the perceptron

    Arguments : 
    X_tain : the train data, numpy array
    y_train : the train target, numpy array with 1 column, with values in {0,1}
    X_val : the validation data, numpy array
    y_val : the validation target, numpy array with 1 column, with values in {0,1}
    n_neurons : numbers of neurons in the first layer
    learning_rate : the model learning rate
    err : the threshold under what you consider the gradient of model is null
    max_iter : the maximum iterations of the gradient descent algorithm (important to stop the training when the model not converge)

    return : The model display the train & val loss, the train & val accuracy
    W, b : the trained parameters
    norm_gradL : the model gradient norm  
    """
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]

    parameters=initialisation(n0, n_neurons, n2)
    activations_train=forward_propagation(X_train, parameters)
    activations_val=forward_propagation(X_val, parameters)
    gradients=back_propagation(X_train,y_train, activations_train, parameters)
    norm_gradL=frobenius_norm_list(list(gradients.values()))
    
    n_iter=1
    train_Loss=[]
    val_Loss=[]
    train_acc=[]
    val_acc=[]
    
    while((norm_gradL>err) & (n_iter<=max_iter)):
        if(n_iter%displaying_step == 0): 

            # computing the train loss
            train_Loss.append(log_loss(activations_train['A2'],y_train))
            y_pred_train=predict(X_train, parameters)
            train_acc.append(accuracy(y_train,y_pred_train))
            
            # computing the validation loss
            val_Loss.append(log_loss(activations_val['A2'],y_val))
            y_pred_val=predict(X_val, parameters)
            val_acc.append(accuracy(y_val,y_pred_val))

            print("train_loss :", train_Loss[-1],
                  "val_loss :", val_Loss[-1],
                  "train_acc :", train_acc[-1],
                  "val_acc :", val_acc[-1],
                  "norm_gradL :", norm_gradL)
        
        # updating the parameters
        parameters=update(gradients, parameters, learning_rate)
        
        # activation
        activations_train=forward_propagation(X_train, parameters)
        activations_val=forward_propagation(X_val, parameters)
        
        # computing the gradients
        gradients=back_propagation(X_train,y_train,activations_train, parameters)
        norm_gradL=frobenius_norm_list(list(gradients.values()))
        n_iter=n_iter+1

    print("n_iter", n_iter)

    # display the loss and the accuracy
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_Loss, label='train_Loss')
    plt.plot(val_Loss, label='val_Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.show()
    
    return (parameters, norm_gradL) 