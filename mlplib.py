import numpy as np
import random as rd
import struct

#Turn the MNIST images into an array of size
#(# of data points x 784), where 784 is the 
#number of pixels
def parse_mnist_images(filename):
  with open(filename, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows*ncols))/255
    #Add the bias input (all 1s)
    bias = np.ones((size, 1))
  return np.hstack((bias, data))

#Parse the MNIST labels into an array of size
#(# of data points x 1)
def parse_mnist_labels(filename):
  with open(filename, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
  return data.reshape(size)

#Parse the MNIST labels into a 1-to-N encoding
#matrix, size (# of data points x 10)
def convert_labels_to_bin(labels, n_outputs):
  bin_labels = np.zeros((len(labels),n_outputs))
  for i in range(0, len(labels)):
    if labels[i] == 0:
      bin_labels[i][0] = 1
    if labels[i] == 1:
      bin_labels[i][1] = 1
    if labels[i] == 2:
      bin_labels[i][2] = 1
    if labels[i] == 3:
      bin_labels[i][3] = 1
    if labels[i] == 4:
      bin_labels[i][4] = 1
    if labels[i] == 5:
      bin_labels[i][5] = 1
    if labels[i] == 6:
      bin_labels[i][6] = 1
    if labels[i] == 7:
      bin_labels[i][7] = 1
    if labels[i] == 8:
      bin_labels[i][8] = 1
    if labels[i] == 9:
      bin_labels[i][9] = 1
  return bin_labels

#Generate random weights in a matrix of size
#(outputs x inputs) with range -cap...cap
def gen_weights(cap, n_outputs, n_inputs):
  return np.random.rand(n_outputs, n_inputs)*2*cap-cap  

#Sigmoid function (1+e^-x)-1
def sigmoid(x):
  return 1/(1+np.exp(-x))

def train(wt_h, wt_o, data, labels, bin_labels, learn_rate, momen):
  #Shuffle order of data points
  data_order = np.arange(0, len(data))
  rd.shuffle(data_order)
 
  #We will use these to save the last iteration's error
  prev_delta_o = 0
  prev_delta_h = 0

  #Iterate through all data points in the set
  for data_pt in data_order:
    #For each data point, calc weighted sum
    wtd_sum_h = np.dot(wt_h, data[data_pt])
    #Check activation using sigmoid
    active_h = sigmoid(wtd_sum_h)
    active_h = np.hstack(([1],active_h))
    #Push the data through to next layer
    wtd_sum_o = np.dot(wt_o, active_h)
    #Check activation using sigmoid
    active_o = sigmoid(wtd_sum_o)

    if np.argmax(active_o) != labels[data_pt]:
      #Calculate the errors based on weighted sum
      err_o = active_o*(1-active_o)*(bin_labels[data_pt]-active_o)
      err_h = active_h*(1-active_h)*np.dot(np.transpose(wt_o), err_o)
      delta_o = learn_rate*np.dot(err_o.reshape((len(err_o),1)), active_h.reshape((1,len(active_h)))) + momen*prev_delta_o
      delta_h = learn_rate*np.dot(err_h[1:].reshape((len(err_h)-1,1)), data[data_pt].reshape((1,len(data[data_pt])))) + momen*prev_delta_h
      wt_h += delta_h
      wt_o += delta_o
      prev_delta_o = delta_o
      prev_delta_h = delta_h
    else:
      prev_delta_o = 0
      prev_delta_h = 0

def test(wt_h, wt_o, data, labels):
  correct = 0
  for data_pt in range(0, len(data)):
    #For each data point, calc weighted sum
    wtd_sum_h = np.dot(wt_h, data[data_pt])
    #Check activation using sigmoid
    active_h = sigmoid(wtd_sum_h)
    active_h = np.hstack(([1],active_h))
    #Push the data through to next layer
    wtd_sum_o = np.dot(wt_o, active_h)
    #Check activation using sigmoid
    active_o = sigmoid(wtd_sum_o)
    if np.argmax(active_o) == labels[data_pt]:
      correct += 1
  return correct/len(data)

def conf_matrix(wt_h, wt_o, data, labels):
  size = len(wt_o)
  cm = np.zeros((size, size))
  for data_pt in range(0, len(data)):
    #For each data point, calc weighted sum
    wtd_sum_h = np.dot(wt_h, data[data_pt])
    #Check activation using sigmoid
    active_h = sigmoid(wtd_sum_h)
    active_h = np.hstack(([1],active_h))
    #Push the data through to next layer
    wtd_sum_o = np.dot(wt_o, active_h)
    #Check activation using sigmoid
    active_o = sigmoid(wtd_sum_o)
    guess = np.argmax(active_o)
    cm[labels[data_pt]][guess] += 1
  return cm
