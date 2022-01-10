import sys, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt


def serializeModel(model, modelName, featureNum):
  initial_type = [('float_input', FloatTensorType([None, featureNum]))]
  onx = convert_sklearn(model, initial_types=initial_type)
  with open(modelName + ".onnx", "wb") as f:
    f.write(onx.SerializeToString())

def modelPredict(modelName, testData):
  sess = rt.InferenceSession(modelName + '.onnx')#load the onnx
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  pred_onx = sess.run([label_name], {input_name: testData.astype(np.float32)})[0]#predict testData
  print(pred_onx)

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target


c45 = DecisionTreeClassifier()
rf = RandomForestClassifier()
lgParam = {'max_iter': 10000}
lg = LogisticRegression(**lgParam)
svmParam = {'C': 8, 'gamma': 0.03125}
svm = SVC(**svmParam)

c45.fit(X, y)
lg.fit(X, y)
rf.fit(X, y)
svm.fit(X, y)

serializeModel(c45, 'c4.5', X.shape[1])
serializeModel(lg, 'LogisticRegression', X.shape[1])
serializeModel(rf, 'RandomForest', X.shape[1])
serializeModel(svm, 'SVM', X.shape[1])

modelPredict('c4.5', X)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

batch_size = 4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(6, 16, 1)
        self.fc1 = nn.Linear(16 * 4 * 1, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

X = X.reshape((150, 1, 4))#reshape the data to make it work in the CNN

net = Net()

idx = np.random.permutation(len(X))#shuffle the index
X = X[idx]#shuffle the data using the new index
y = y[idx]#shuffle the answer
x_train = X[:100, :]#get the training set
y_train = y[:100]
x_test = X[100:, :]#get the testing set
y_test = y[100:]
x_train = torch.from_numpy(x_train)#transform the data from numpy to torch
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()
train_dataset = TensorDataset(x_train, y_train)#prepare the training data
test_dataset = TensorDataset(x_test, y_test)

trainloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dummny_channel = 1
dummy_feature = 4
dummy_input = torch.randn(batch_size, dummny_channel, dummy_feature)
torch.onnx.export(net,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "cnn.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 

modelPredict('cnn', x_test.numpy())


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 1, 4)
            nn.Conv1d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=2,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 1, 2)
            nn.ReLU(),    # activation
            nn.MaxPool1d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 1, 2)
            nn.Conv1d(16, 32, 2, 1, 1),  # output shape (32, 1, 2)
            nn.ReLU(),  # activation
            nn.MaxPool1d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 1 * 1, 3)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()


for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dummy_input = torch.randn(batch_size, dummny_channel, dummy_feature)
torch.onnx.export(net,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "cnn2.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 

modelPredict('cnn2', x_test.numpy())
