import numpy as np
from numpy.core.numeric import indices
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    W_shape, b_shape = (in_size, out_size), out_size
    W = np.random.uniform(-np.sqrt(6/(in_size + out_size)), np.sqrt(6/(in_size + out_size)), W_shape)
    b = np.zeros(b_shape)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    res = 1/(1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros(x.shape)
    for i in range(x.shape[0]):
        ex = x[i,:]
        ex -= np.max(ex)
        num = np.exp(ex)
        den = np.sum(num)
        res[i,:] = (num*1.)/den    

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    cnt = 0
    for i in range(probs.shape[0]):
        pred = np.argmax(probs[i,:])
        if y[i, pred] == 1:
            cnt += 1
    
    acc = (cnt*1.) / probs.shape[0]
    loss = -np.sum(y * np.log(probs))

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    num,_ = delta.shape
    gpa = activation_deriv(post_act)
    grad_W = np.matmul(X.T, gpa * delta)
    grad_X = np.matmul(gpa * delta, W.T)
    grad_b = np.matmul(np.ones((1, num)), gpa * delta)
    grad_b = grad_b.flatten()
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]


def get_random_batches(x,y,batch_size):
    batches = []
    num,_ = x.shape
    indices = np.random.choice(num,size = ((num // batch_size),batch_size))
    for i in range(len(indices)):
        batches.append((x[indices[i],:], y[indices[i],:]))
    return batches