## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input shape: 1, 224, 224
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128,3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024,3)
        
#         self.dropout1 = nn.Dropout(0.10)
#         self.dropout2 = nn.Dropout(0.20)
#         self.dropout3 = nn.Dropout(0.30)
#         self.dropout4 = nn.Dropout(0.40)
#         self.dropout5 = nn.Dropout(0.45)
#         self.dropout6 = nn.Dropout(0.50)
        self.dropout7 = nn.Dropout(0.30)
        self.dropout8 = nn.Dropout(0.30)

        self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU() # use F.relu()
        # self.elu = nn.ELU() # use F.elu()
        
        self.fc1 = nn.Linear(1024*1*1, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x))) # output shape: 32, 111, 111
#         x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x))) # output shape: 64, 54, 54
#         x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x))) # output shape: 128, 26, 26
#         x = self.dropout3(x)
        x = self.pool(F.relu(self.conv4(x))) # output shape: 256, 12, 12
#         x = self.dropout4(x)
        x = self.pool(F.relu(self.conv5(x))) # output shape: 512, 5, 5
#         x = self.dropout5(x)
        x = self.pool(F.relu(self.conv6(x))) # output shape: 1024, 1, 1
#         x = self.dropout6(x)
        # x = torch.flatten(x) # size mismatch
        x = x.view(-1, x.size(1))
        x = F.relu(self.fc1(x))
        x = self.dropout7(x)
        x = F.relu(self.fc2(x))
        x = self.dropout8(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
