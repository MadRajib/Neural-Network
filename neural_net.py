#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy
import scipy.special
import matplotlib.pyplot
# get_ipython().run_line_magic('matplotlib', 'inline')
#neural network class definition
class neuralNetwork:
    
    
    #initialise the neural network
    def __init__(self,ipnodes,hiddennodes,opnodes,learningrate):
        
        #set number of nodes in eache i/p,hidden and o/p layer
        self.inodes = ipnodes
        self.hnodes = hiddennodes
        self.onodes = opnodes
        
        #learning rate
        self.lr = learningrate
        
        #link weight matrices, wih and who
        #weights inside the arrays are w_i_j, where link is from node i to node j of next layer
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        #activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    
    #train the neural network
    def train(self,input_list,targets_list):
        targets = numpy.array(targets_list,ndmin=2).T
        
        inputs = numpy.array(input_list,ndmin=2).T
        
        
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        #error is the (target -actual)
        output_errors = targets - final_outputs
        
        #hidden layer error is the output_errors,split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        #update the weights for the links between th hidden and output layers
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 -hidden_outputs)), numpy.transpose(inputs))
        pass
    
    #query the neural network
    def query(self,inputs_list):
        #convert input list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        
        
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# In[83]:


# number of i/p,hidden  and o/p nodes
ip_nodes = 784
h_nodes = 100
op_nodes = 10

#learning rate is 0.3
lr_rate = 0.1

#create instance of neural network
n = neuralNetwork(ip_nodes,h_nodes,op_nodes,lr_rate)


# In[85]:


#create mnist training data csv file into a list
with open("minst_dataset/mnist_train.csv",'r') as data_file:
    training_data_list = data_file.readlines()


# In[86]:


# train the neural network

#epochs is the number of times the training data set is used for training
epochs = 5
for e in range(epochs):
    #go through all the records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        #scale and shift the i/p
        scaled_input = (numpy.asfarray(all_values[1:])/255.0 * 0.99) +0.01

        #create the target output value,max=0.99 and min=0.01
        targets = numpy.zeros(op_nodes)+0.01
        #all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        #train the network
        n.train(scaled_input,targets)


# In[87]:


#testing the network model

#get the test data_set from the file
with open("minst_dataset/mnist_test.csv",'r') as data_file:
    test_data_list = data_file.readlines()
    
#test the neural network based on scorecard
scorecard = []

#go through all the records in the test data set
for record in test_data_list:
    all_values = record.split(',')
    correct_lable = int(all_values[0])
    
    #correct answer
    #print(correct_lable,"correct lable")
    
    #query and output 
    outputs = n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
    
    #get the index of highest value 
    lable = numpy.argmax(outputs)
    #print(lable,"network answer")
    
    #append correct or incorrect to list
    if(lable == correct_lable):
        scorecard.append(1)
    else:
        scorecard.append(0)
    


# In[89]:


#calculate the performance score
scorecard_array = numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)


# In[ ]:




