## TODO: define the convolutional neural network architecture

import torch
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
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64,4 )
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool=nn.MaxPool2d(2,2)
        
        self.fc=nn.Linear(43264,1024)
        self.bn = nn.BatchNorm1d(1024)
        
        self.fc2=nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.25)
        self.dp3 = nn.Dropout(0.35)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.55)
        self.dp6 = nn.Dropout(0.6)

        self.out=nn.Linear(512,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        print(x.shape)
        x = x.reshape(x.shape[0],-1)
        print(x.shape)
        x = self.dp5(F.relu(self.fc(x)))
        x = self.dp6(F.relu(self.fc2(x)))
        x = self.out(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
