import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

#----------------Model Architecture-----------------
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=6): 
        super().__init__()
        #Use a 4-layer CNN architecture optimized for understanding facial feature patterns 
        #Convolutional layers extract deep features while managing GPU computational load 
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.batchnorm1 = torch.nn.BatchNorm2d(64)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm2d(128)
        self.batchnorm4 = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = torch.nn.Linear(256 * 7 * 7, 2048) 
        self.bn_fc1 = torch.nn.BatchNorm1d(2048)
        self.fc2 = torch.nn.Linear(2048, 512)
        self.bn_fc2 = torch.nn.BatchNorm1d(512)
        self.fc3 = torch.nn.Linear(512, num_classes)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)
#Setup device and class labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] 
#Load the model
model = ConvNet(num_classes=len(classes)).to(device)

model_path = 'best_model.pth'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#Check if the output layer size matches the current class count
    if checkpoint['fc3.weight'].shape[0] != len(classes):

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)
else:
    print("No Model")

model.eval()
#-----------------Real-time Preprocessing----------------------------
#Must match the original training input specifications (48x48 Grayscale)
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
#--------------------------------OpenCV Face Detection Pipeline-----------------------------------
#Use Haar Cascade for efficient real-time face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
#Convert frame to grayscale for the face detector
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        #Extract the Region of Interest (ROI) containing the face
        face_roi = frame[y:y+h, x:x+w]
        #Convert BGR (OpenCV) to RGB (PIL) to ensure compatibility with torchvision transforms
        img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = classes[predicted.item()]
        #Draw results on the live video feed    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()