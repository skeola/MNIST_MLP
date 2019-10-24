import csv
import matplotlib.pyplot as plt
import mlplib as mlp
import numpy as np

#Constants
N_INPUTS = 784
N_HIDDEN = 20
N_OUTPUTS = 10
LEARN_RATE = 0.1
EPOCHS = 5
MOMENTUM = 0.9

#Get the data from the MNIST database
train_data = mlp.parse_mnist_images("input/train-images-idx3-ubyte")
test_data = mlp.parse_mnist_images("input/t10k-images-idx3-ubyte")
train_labels_temp = mlp.parse_mnist_labels("input/train-labels-idx1-ubyte")
test_labels_temp = mlp.parse_mnist_labels("input/t10k-labels-idx1-ubyte")
train_labels = mlp.convert_labels_to_bin(train_labels_temp, N_OUTPUTS)
test_labels = mlp.convert_labels_to_bin(test_labels_temp, N_OUTPUTS)

#Randomly initialize weights between -0.05 and 0.05
weights_to_hidden = mlp.gen_weights(0.05, N_HIDDEN, N_INPUTS + 1)
weights_to_output = mlp.gen_weights(0.05, N_OUTPUTS, N_HIDDEN + 1)

with open("output/accuracy-lr-"+str(LEARN_RATE)+"-nhid-"+str(N_HIDDEN)+".csv", mode='w') as data_file:
  data_file = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  data_file.writerow(['epoch', 'train acc', 'test acc'])
  data_file.writerow([0,mlp.test(weights_to_hidden, weights_to_output, train_data, train_labels_temp),mlp.test(weights_to_hidden, weights_to_output, test_data, test_labels_temp)])
  
  for i in range(1, EPOCHS+1):
    mlp.train(weights_to_hidden, weights_to_output, train_data, train_labels_temp, train_labels, LEARN_RATE, MOMENTUM)
    train_acc = mlp.test(weights_to_hidden, weights_to_output, train_data, train_labels_temp)
    test_acc = mlp.test(weights_to_hidden, weights_to_output, test_data, test_labels_temp)
    print("EPOCH ", i, ": ",train_acc,", ",test_acc)
    data_file.writerow([i, train_acc, test_acc])

cm = mlp.conf_matrix(weights_to_hidden, weights_to_output, test_data, test_labels_temp)

with open("output/conf-matrix-lr-"+str(LEARN_RATE)+"-nhid-"+str(N_HIDDEN)+".csv", mode='w') as conf_file:
  conf_file = csv.writer(conf_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  conf_file.writerow([' ','0','1','2','3','4','5','6','7','8','9'])
  for i in range(0,10):
    conf_file.writerow([i,cm[i][0],cm[i][1],cm[i][2],cm[i][3],cm[i][4],cm[i][5],cm[i][6],cm[i][7],cm[i][8],cm[i][9]])
