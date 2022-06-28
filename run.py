import cv2
# from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
# from __future__ import print_function
import argparse
# import numpy  as np
# from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
net = Deep_Emotion()  # In pytorch we have to first load the model
# file_name = 'C:/Users/hp/OneDrive/Documents/Malay_Projects/SE Project/Deep-Emotion/deep_emotion-100-128-0.005.pt'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_name = 'deep_emotion-100-128-0.005.pt'
net.load_state_dict(torch.load(file_name))
net.to(device)

path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# SET THE RECTANGLE BACKGROUND TO WHITE
rectangle_bgr = (255, 255, 255)

# MAKE A BLACK IMAGE
img = np.zeros((500, 500))

# SET SOME TEXT
text = "Some text in a box!"

# Get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(
    text, font, fontScale=font_scale, thickness=1)[0]

# Set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25

# make the coords of the box with a small padding of 2 pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
                                               text_width + 2, text_offset_y - text_width - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font,
            fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in facces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]  # CROPPING THE FACE
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                final_image = cv2.resize(gray, (48, 48))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0
                dataa = torch.from_numpy(final_image)
                dataa = dataa.type(torch.FloatTensor)
                dataa = dataa.to(device)
                outputs = net(dataa)
                pred = F.softmax(outputs, dim=1)
                prediction = mapping[int(torch.argmax(pred))]
                print('prediction = ' + prediction)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                # HELLO
                font = cv2.FONT_HERSHEY_PLAIN
