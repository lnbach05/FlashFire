
# Train an autopilot for autonomous ground vehicle using
# convolutional neural network and labeled images. 


import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import cnn_network
import cv2 as cv

# Pass in command line arguments for path name 
if len(sys.argv) != 2:
    print(f'Training script needs 1 parameters!!!')
    sys.exit(1) #Exit with an error code
else:
    data_datetime = sys.argv[1]
    # model_name = sys.argv[2]
    # figure_name = sys.argv[3]
    


# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


class BearCartDataset(Dataset): 
    """
    Customized dataset
    """
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        image_tensor = self.transform(image)
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        return image.float(), steering, throttle



def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target= im.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss= loss_fn(pred, target)
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples += target.shape[0]
        print(f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader)}]")
        ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    # losses_train.append(ep_loss)
    return ep_loss

    # size = len(dataloader.dataset)
    # model.train()
    # epoch_loss = 0.0
    #
    # for batch, (image, steering, throttle) in enumerate(dataloader):
    #     # Combine steering and throttle into one tensor (2 columns, X rows)
    #     target = torch.stack((steering, throttle), -1) 
    #     X, y = image.to(DEVICE), target.to(DEVICE)
    #
    #     # Compute prediction error
    #     pred = model(X)  # forward propagation
    #     batch_loss = loss_fn(pred, y)  # compute loss
    #     optimizer.zero_grad()  # zero previous gradient
    #     batch_loss.backward()  # back propagatin
    #     optimizer.step()  # update parameters
    #     
    #     batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
    #     epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
    #     print(f"loss: {batch_loss:>7f} [{sample_count:>5d}/{size:>5d}]")
    #     
    # return epoch_loss

        

def test(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.
    with torch.no_grad():
        for b, (im, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target= im.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
        # losses_eval.append(ep_loss_eval)
    return ep_loss
    # # Define a test function to evaluate model performance
    #
    # #size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    # model.eval()
    # test_loss = 0.0
    # with torch.no_grad():
    #     for image, steering, throttle in dataloader:
    #         #Combine steering and throttle into one tensor (2 columns, X rows)
    #         target = torch.stack((steering, throttle), -1) 
    #         X, y = image.to(DEVICE), target.to(DEVICE)
    #         pred = model(X)
    #         test_loss += loss_fn(pred, y).item()
    # test_loss /= num_batches
    # print(f"Test Error: {test_loss:>8f} \n")
    #
    # return test_loss



# MAIN
# Create a dataset
data_dir = os.path.join(sys.path[0], 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')  # the name of the csv file
img_dir = os.path.join(data_dir, 'images') # the name of the folder with all the images in it
bearcart_dataset = BearCartDataset(annotations_file, img_dir)
print(f"data length: {len(bearcart_dataset)}")
# Create training dataloader and test dataloader
train_size = round(len(bearcart_dataset)*0.9)
test_size = len(bearcart_dataset) - train_size 
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)
# Create model
model = cnn_network.megaNet().to(DEVICE)  # choose the architecture class from cnn_network.py
# Hyper-parameters (lr=0.001, epochs=10 | lr=0.0001, epochs=15 or 20)
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.05)  # Adjust the step_size and gamma as needed
loss_fn = nn.MSELoss()
epochs = 15
# # Optimize the model
# train_loss = []
# test_loss = []
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     training_loss = train(train_dataloader, model, loss_fn, optimizer)
#     testing_loss = test(test_dataloader, model, loss_fn)
#     print("average training loss: ", training_loss)
#     print("average testing loss: ", testing_loss)
#     # Apply the learning rate scheduler after each epoch
#     #scheduler.step()
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Learning rate after scheduler step: {current_lr}")
#     # save values
#     train_loss.append(training_loss)
#     test_loss.append(testing_loss)   
#
# print(f"Optimize Done!")


#
#
# #print("final test lost: ", test_loss[-1])
# len_train_loss = len(train_loss)
# len_test_loss = len(test_loss)
# print("Train loss length: ", len_train_loss)
# print("Test loss length: ", len_test_loss)
#
#
# # create array for x values for plotting train
# epochs_array = list(range(epochs))
#
# # Graph the test and train data
# plot_title = f'{model._get_name()} - {epochs} pochs - {lr} learning rate'
# fig = plt.figure()
# axs = fig.add_subplot(1,1,1)
# plt.plot(epochs_array, train_loss, color='b', label="Training Loss")
# plt.plot(epochs_array, test_loss, '--', color='orange', label='Testing Loss')
# axs.set_ylabel('Loss')
# axs.set_xlabel('Training Epoch')
# axs.set_title('Analyzing Training and Testing Loss')
# axs.legend()
# fig.savefig(model_path + figure_name)
#
# # Save the model
# torch.save(model.state_dict(), model_path + model_name)
#
#
#
