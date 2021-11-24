import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
train_y = train_data['train_labels']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, hidden_size, params, 'l1')
initialize_weights(hidden_size, hidden_size, params, 'h1')
initialize_weights(hidden_size, hidden_size, params, 'h2')
initialize_weights(hidden_size, 1024, params, 'out')

params['m_Wl1'], params['m_Wh1'], params['m_Wh2'], params['m_Wout']  = np.zeros((1024,hidden_size)), np.zeros((hidden_size,hidden_size)), np.zeros((hidden_size,hidden_size)), np.zeros((hidden_size,1024))
params['m_bl1'], params['m_bh1'], params['m_bh2'], params['m_bout']  = np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(1024)

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        h1 = forward(xb, params, 'l1', relu)
        h2 = forward(h1, params, 'h1', relu)
        h3 = forward(h2, params, 'h2', relu)
        output = forward(h3, params, 'out', sigmoid)
        
        loss = np.sum((xb - output)**2)
        total_loss += loss
        delta = 2*(output - xb)
        delta1 = backwards(delta, params, 'out', sigmoid_deriv)
        delta2 = backwards(delta1, params, 'h2', relu_deriv)
        delta3 = backwards(delta2, params, 'h1', relu_deriv)
        _ = backwards(delta3, params, 'l1', relu_deriv)

        # params['W' + 'out'] -= learning_rate * params['grad_W' + 'out']
        # params['b' + 'out'] -= learning_rate * params['grad_b' + 'out']
        # params['W' + 'h2'] -= learning_rate * params['grad_W' + 'h2']
        # params['b' + 'h2'] -= learning_rate * params['grad_b' + 'h2']
        # params['W' + 'h1'] -= learning_rate * params['grad_W' + 'h1']
        # params['b' + 'h1'] -= learning_rate * params['grad_b' + 'h1']
        # params['W' + 'l1'] -= learning_rate * params['grad_W' + 'l1']
        # params['b' + 'l1'] -= learning_rate * params['grad_b' + 'l1']
        
        #momentum
        for k, v in params.items():
            if '_' in k:
                continue

            params['m_{}'.format(k)] = (0.9*params['m_{}'.format(k)]) - (learning_rate*params['grad_{}'.format(k)])
            v += params['m_{}'.format(k)]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
h1 = forward(train_x, params, 'l1', relu)
h2 = forward(h1, params,'h1', relu)
h3 = forward(h2, params,'h2', relu)
out = forward(h3, params,'out', sigmoid)

classes = [0, 1, 2] #for visualization
for c in classes:
    x_c, y_c = [], []
    for i in range(len(train_y)):
        if np.argmax(train_y[i]) == c:
            x_c.append(np.reshape(train_x[i], (32,32)).T)
            y_c.append(np.reshape(out[i], (32,32)).T)
    
    fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize=(2,3))
    ax[0,0].imshow(x_c[0], cmap = 'gray')
    ax[0,1].imshow(y_c[0], cmap = 'gray')
    ax[1,0].imshow(x_c[1], cmap = 'gray')
    ax[1,1].imshow(y_c[1], cmap = 'gray')
    ax[2,0].imshow(x_c[2], cmap = 'gray')
    ax[2,1].imshow(y_c[2], cmap = 'gray')
    plt.show()
    plt.clf()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
