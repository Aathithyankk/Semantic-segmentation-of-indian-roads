import os
import numpy as np
import cv2
import torch
import torchvision #images and videos
import torch.nn as nn # for functions in neural network
from tqdm import tqdm # to look for the progress
import random 
import torchvision.transforms as transforms #data augmentation
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
%matplotlib inline


PATH = "../input/semantic-segmentation-datasets-of-indian-roads/Indian_road_data/Indian_road_data"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from np to tensor format

TRANSFORMS = transforms.Compose([transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

labels = [0, 1, 2, 3]

def dataset_generator() :
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    
    data = ["train", "val"]
    labels = [0, 1, 2, 3]
    
    for i in data:
        for j in os.listdir(PATH + "/Raw_images/" + str(i)) :
            for k in os.listdir(PATH + "/Raw_images/" + str(i) + "/" + str(j)) :
                cntn = random.randint(0,3)
                if cntn :
                    continue
                
                # Raw_images
                img = cv2.imread(PATH + "/Raw_images/" + str(i) + "/" + str(j) + "/" + str(k))
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_CUBIC)
                img = np.uint8(img)
                if i == "train" :
                    train_images.append(img)
                else :
                    valid_images.append(img)
                
                    
                # Masks
                masks = []
                mask = cv2.imread(PATH + "/Masks/" + str(i) + "/" + str(j) + "/" + str(k).replace("leftImg8bit.jpg", "gtFine_labelTrainIds.png"))
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_CUBIC)
                
                for label in labels: 
                    msk = (mask==label)*255
                    masks.append(msk)
                if i == "train" :
                    train_labels.append(masks)
                else :
                    valid_labels.append(masks)
                    
            print("Image and Masks Extraction for " + str(j) + " is done...")
                    
    train_images = np.array(train_images)
    valid_images = np.array(valid_images)
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    
    return train_images, valid_images, train_labels, valid_labels


class create_dataset(Dataset) :
    def __init__(self, images, masks, transforms = None) :
        self.images = images
        self.labels = masks
        self.transforms = transforms
        
    def __len__(self) :
        return len(self.images)
        
    def __getitem__(self, index) :
        img = self.images[index]
        masks = self.labels[index]
        
        if self.transforms is not None :
            img = self.transforms(img)
            masks = self.transforms(masks)

        return img, masks

    

def load_dataset() :
    print("Dataset Generation started...")
    train_images, valid_images, train_labels, valid_labels = dataset_generator()
    
    print("Dataset Generated...")
    print(train_images.shape, valid_images.shape, train_labels.shape, valid_labels.shape)
    
    train_dataset = create_dataset(train_images, train_labels, TRANSFORMS)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = BATCH_SIZE, 
                                               shuffle = True, 
                                               num_workers = 1)
    print("Step 1 Crossed...")
    
    valid_dataset = create_dataset(valid_images, valid_labels, TRANSFORMS)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True,
                                               num_workers = 1)
    print("Step 2 Crossed...")
    
    for i, j in train_loader:
        print(i.shape)
        break
    
    return train_loader, valid_loader


train_loader, valid_loader = load_dataset()
print("Dataset Loaded Successfully...")


model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = True)
model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 4)
model.to(device)

criterion = nn.MSELoss(reduction = "mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, 1+EPOCHS) :
    train_running_loss = 0
    valid_running_loss = 0
    
    model.train()
    
    pbar = tqdm(iter(train_loader), desc=f"Epoch {epoch}/{EPOCHS}: ")
    
    for idx, (data_, labels_) in enumerate(valid_loader) :
        data_, labels_ = data_.type(torch.FloatTensor), labels_.type(torch.FloatTensor)
        data_, labels_ = data_.to(device), labels_.to(device)
        
        optimizer.zero_grad()
        outputs = model(data_)
        loss = criterion(outputs["out"], labels_)
        loss.backward()
        optimizer.step() #optimizer read
        
        train_running_loss += loss.item() 
        
        pbar.set_postfix(loss=loss.item())
        
    with torch.no_grad():
        model.eval()
        for idx, (data_, labels_) in enumerate(pbar) :
            data_, labels_ = data_.type(torch.FloatTensor), labels_.type(torch.FloatTensor)
            data_, labels_ = data_.to(device), labels_.to(device)

            optimizer.zero_grad()
            outputs = model(data_)
            loss = criterion(outputs["out"], labels_)
            
            valid_running_loss += loss.item()
    
    print(f"Train_loss = {(train_running_loss/len(train_loader)):.4f} ;  Valid_loss = {(valid_running_loss/len(valid_loader)):.4f}")

