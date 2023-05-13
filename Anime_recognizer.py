import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor, transforms
from torchvision.transforms import Normalize
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import streamlit as st
import numpy as np
import os
from PIL import Image


# xyz = st.file_uploader("Upload a jpg file", type="jpg")

st.title('Anime Image to Anime Name')

# Create an image upload widget
xyz = st.file_uploader('Upload an image', type=['jpg'])

# Check if an image was uploaded
if xyz is not None:
    # Open and display the uploaded image
    image = Image.open(xyz)
    st.image(image, caption='Uploaded Image')
else:
    st.write('Please upload an image.')


# Define the CNN model
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1, stride = 1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1, stride = 1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 16 * 16, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=6)
        
        # Define the activation function
        self.relu = nn.ReLU()
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        # Define dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)
        
    def forward(self, x):
        # Perform the forward pass through the convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 16 * 16)
        
        # Perform the forward pass through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# Initialize the model and the optimizer
model = MyCNN(num_classes=6)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
if xyz:

    filename = xyz
    image = Image.open(filename)
    image = transform(image)
    image = image.unsqueeze(0)


    classes = ['Grave of the Fireflies','I want to eat your Pancreas','Perfect Blue','Spirited Away','Weathering With You','Your Name']

    model = MyCNN(num_classes = 6)

    # Load the trained weights into the model
    model.load_state_dict(torch.load("C:\Site\my_model2_as_dictionary_drop(0.5,0.7).pt"))
    # Get the predicted class probabilities
    with torch.no_grad():
        outputs = model(image)

    # Convert the predicted class probabilities to class labels
    _, predicted = torch.max(outputs.data, 1)

def main():
    if st.button("Predict"):
        st.write("Predicted class label:", classes[predicted.item()])

if __name__ == "__main__":
    main()




