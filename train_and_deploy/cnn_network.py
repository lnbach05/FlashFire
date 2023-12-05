import torch.nn as nn


class hblNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        
        #Convolution Layers
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
        # self.conv_3 = nn.Conv2d(64, 82, kernel_size=(3, 3), stride=(2, 2))
        # self.conv_4 = nn.Conv2d(48, 60, kernel_size=(3, 3), stride=(2, 2))
        # self.conv_5 = nn.Conv2d(60, 72, kernel_size=(3, 3), stride=(1, 1))
        # self.conv_6 = nn.Conv2d(72, 84, kernel_size=(3, 3), stride=(1, 1))

        #Compute Sizes
        size_input = (width, height)  # output size: (input_size + 2*padding_size - kernel_size) / stride_size + 1
        size_1 = (int((size_input[0] - 5) / 2 + 1), int((size_input[1] - 5) / 2 + 1))
        size_2 = (int((size_1[0] - 5) / 2 + 1), int((size_1[1] - 5) / 2 + 1))
        # size_3 = (int((size_2[0] - 3) / 2 + 1), int((size_2[1] - 3) / 2 + 1))
        # size_4 = (int((size_3[0] - 3) / 2 + 1), int((size_3[1] - 3) / 2 + 1))
        # size_5 = (int(size_4[0] - 3 + 1), int(size_4[1] - 3 + 1))
        # size_6 = (int(size_5[0] - 3 + 1), int(size_5[1] - 3 + 1))
        size_fc_input = size_2[0] * size_2[1] * 64

        #Fully Connected Layers
        self.fc1 = nn.Linear(size_fc_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):              
        x = self.relu(self.conv_1(x))  
        x = self.relu(self.conv_2(x))  
        # x = self.relu(self.conv_3(x))  
        # x = self.relu(self.conv_4(x))  
        # x = self.relu(self.conv_5(x)) 
        # x = self.relu(self.conv_6(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DonkeyNet(nn.Module):
    """
    Dokney docs --> parts --> keras and fastai (uses PyTorch)
    donkeycar (github) --> donkeycar --> parts --> keras.py and fastai.py (try fastai first)
    https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py
    """

    def __init__(self, width, height):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv_2 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        # Comupte sizes
        size_input = (width, height)  # output size: (input_size + 2*padding_size - kernel_size) / stride_size + 1
        size_1 = (int((size_input[0] - 5) / 2 + 1), int((size_input[1] - 5) / 2 + 1))
        size_2 = (int((size_1[0] - 5) / 2 + 1), int((size_1[1] - 5) / 2 + 1))
        size_3 = (int((size_2[0] - 5) / 2 + 1), int((size_2[1] - 5) / 2 + 1))
        size_4 = (int(size_3[0] - 3 + 1), int(size_3[1] - 3 + 1))
        size_5 = (int(size_4[0] - 3 + 1), int(size_4[1] - 3 + 1))
        size_fc_input = size_5[0] * size_5[1] * 64

        self.fc1 = nn.Linear(size_fc_input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):              #  120x160 images
        x = self.relu(self.conv_1(x))  # (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv_2(x))  # (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv_3(x))  # (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv_4(x))  # 12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv_5(x))  # 10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

h = hblNet(120, 160)
d = DonkeyNet(120, 160)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("HBL NET: ", count_parameters(h))
print("Donkey NET: ", count_parameters(d))



