import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from nn import *

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 2e-3
hidden_size = 64
##########################
##### your code here #####  
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')

ys, preds = [], []
train_losses, train_acc, val_losses, val_acc = [], [], [], []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    preds, ys = [], []
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        delta1 = probs - yb

        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        
        # apply gradient
        params['W' + 'output'] -= learning_rate * params['grad_W' + 'output']
        params['b' + 'output'] -= learning_rate * params['grad_b' + 'output']
        params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1']

        preds.extend(probs)
        ys.extend(yb)

    total_loss /= len(batches)
    total_acc /= len(batches)
    train_losses.append(total_loss)
    train_acc.append(total_acc)

    valid_acc = None
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    val_losses.append(valid_loss)
    val_acc.append(valid_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

plt.plot(np.arange(max_iters),train_losses,'r')
plt.plot(np.arange(max_iters),val_losses,'b')
plt.legend(['training loss','valid loss'])
plt.show()
plt.clf()
plt.plot(np.arange(max_iters),train_acc,'r')
plt.plot(np.arange(max_iters),val_acc,'b')
plt.legend(['training loss','valid loss'])
plt.show()
plt.clf()
    
# run on validation set and report accuracy! should

# if True: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
W_firstlayer = params['Wlayer1']
shape,cols = W_firstlayer.shape
fig1 = plt.figure()
grid = ImageGrid(fig1, 111, nrows_ncols=(8,8,),axes_pad=0.0)
for i in range(cols):
    grid[i].imshow(W_firstlayer[:,i].reshape((32,32)))


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# confusion_matrix[preds[]]
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()