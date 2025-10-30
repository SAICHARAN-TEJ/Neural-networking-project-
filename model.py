import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognitionNN(nn.Module):
    """
    Neural Network for handwritten digit recognition
    Architecture: Input(784) -> Hidden(128) -> Hidden(64) -> Output(10)
    """
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, num_classes=10):
        super(DigitRecognitionNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        return x

class AdvancedDigitCNN(nn.Module):
    """
    Advanced CNN for better accuracy
    """
    def __init__(self, num_classes=10):
        super(AdvancedDigitCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x