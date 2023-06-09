import streamlit as st
import leafmap.foliumap as leafmap
from PIL import Image
from io import StringIO

import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
matplotlib.style.use('ggplot')
import torch
import cv2

import numpy as np
import glob as glob
import os
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
import requests
from streamlit_lottie import st_lottie

def load_animation(url):
    r=requests.get(url)
    return r.json()




import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
a=SRCNN()

# predict the high resolution image

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path,0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        img = cv2.resize(img, (64, 64), cv2.INTER_CUBIC)
        img = img.astype('float32') / 255
        # img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        return img
    except Exception as e:
        print(repr(e))

# upload a low resolution image and a high resolution image
low_res_image_path = "Bioimaging_data3\A0.3\\103064_0.3.jpg"
high_res_image_path = "Bioimaging_data3\A0.7\\103064_0.7.jpg"

# preprocess the low resolution image (/content/drive/MyDrive/Bioimaging_data/Test images/1050073_0.3.jpg)
low_res_image = preprocess_image(low_res_image_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model = SRCNN().to(device)
saved_model.load_state_dict(torch.load('Bioimaging_data3/model.pth',map_location=torch.device('cpu')))

with torch.no_grad():
    saved_model.eval()
    output = saved_model(low_res_image.to(device))
    output = output.cpu()
    output = output.numpy()
    output = output.squeeze()
    output = (output + 1) / 2 * 255
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

# preprocess the high resolution image
high_res_image = preprocess_image(high_res_image_path)
high_res_image = high_res_image.numpy()
high_res_image = high_res_image.squeeze()
high_res_image = (high_res_image + 1) / 2 * 255
high_res_image = np.clip(high_res_image, 0, 255)
high_res_image = high_res_image.astype('uint8')

# preprocess the low resolution image for display
low_res_image_display = cv2.imread(low_res_image_path)
low_res_image_display = cv2.cvtColor(low_res_image_display, cv2.COLOR_BGR2RGB)
# low_res_image_display = cv2.resize(low_res_image_display, (256, 256), cv2.INTER_CUBIC)

# preprocess the predicted image for display
output_display = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
# output_display = cv2.resize(output_display, (256, 256), cv2.INTER_CUBIC)


# plot the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(low_res_image_display)
axs[0].set_title('Low Resolution')
# axs[2].imshow(high_res_image,cmap="gray")
# axs[2].set_title('High Resolution')
axs[1].imshow(output_display)
axs[1].set_title('Predicted')
plt.show()




def app():

    st.title("Super Resolution Convolutional Neural Network")
    lottie=load_animation("https://assets5.lottiefiles.com/private_files/lf30_8npirptd.json")

    st_lottie(lottie,height=300,width=400,quality='high',key="initial")

    st.subheader("Our project aims to demonstrate that the traditional pipeline used for single-image super-resolution is equivalent to a deep convolutional neural network. Motivated by this insight, we propose a model that utilizes a convolutional neural network to directly learn the mapping between low- and high-resolution images in an end-to-end manner. The method used in our project differs from existing approaches as it does not rely on explicitly learning dictionaries or manifolds to model the patch space. Instead, these are implicitly achieved through hidden layers. Additionally, our approach formulates patch extraction and aggregation as convolutional layers, which are incorporated into the optimization process. With our method, the entire super-resolution pipeline is learned through training, with minimal pre- or post-processing required. We refer to our proposed model as the Super-Resolution Convolutional Neural Network (SRCNN).")
    image=Image.open("outpucnn.png")
    st.image(image,caption="CNN",use_column_width=True)

    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        print(uploaded_file.name)
        st.image(uploaded_file,caption="Uploaded Image")
        bytes_data = uploaded_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)
        low_res_image = cv2.resize(img, (64, 64), cv2.INTER_CUBIC)
        low_res_image = low_res_image.astype('float32') / 255
        low_res_image = np.expand_dims(low_res_image, axis=0)
        low_res_image = torch.from_numpy(low_res_image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            saved_model.eval()
            output = saved_model(low_res_image.to(device))
            output = output.cpu()
            output = output.numpy()
            output = output.squeeze()
            output = (output + 1) / 2 * 255
            output = np.clip(output, 0, 255)
            output = output.astype('uint8')

        low_res_image_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_display = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(low_res_image_display)
        axs[0].set_title('Low Resolution')
        # axs[2].imshow(high_res_image,cmap="gray")
        # axs[2].set_title('High Resolution')
        axs[1].imshow(output_display)
        axs[1].set_title('Predicted')

        ok2 = st.button("See Prediction")
        if ok2:
            st.pyplot(fig)







