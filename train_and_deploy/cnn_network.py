import torch.nn as nn


class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv64 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
        self.conv128 = nn.Conv2d(16, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv256 = nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))

        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / stride) + 1

        self.fc1 = nn.Linear(32*70*70, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        

    def forward(self, x):               
        x = self.relu(self.conv64(x))  
        x = self.relu(self.conv128(x))  
        x = self.relu(self.conv256(x))  

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class moderateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv48 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1))
        self.conv64 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1))

        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / (stride + 1)


        self.fc1 = nn.Linear(64*68*68, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):         
        x = self.relu(self.conv24(x))        
        x = self.relu(self.conv32(x))  
        x = self.relu(self.conv48(x))  
        x = self.relu(self.conv64(x))  

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class megaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv16 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
        self.conv24 = nn.Conv2d(16, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2))
        self.conv48 = nn.Conv2d(32, 48, kernel_size=(2, 2), stride=(1, 1))
        self.conv64 = nn.Conv2d(48, 64, kernel_size=(1, 1), stride=(1, 1))


        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / (stride + 1)

        #200x200
        #((200 - 5) / 2) + 1 = 98
        #((98 - 5) / 2) + 1 = 47
        #((47 - 3)) / 2) + 1 = 23
        #((23 - 2) / 1) + 1 = 22
        #((22 - 1) / 1) + 1 = 22


        self.fc1 = nn.Linear(64*34*34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               
        x = self.relu(self.conv16(x))  
        x = self.relu(self.conv24(x))  
        x = self.relu(self.conv32(x))
        x = self.relu(self.conv48(x))  
        x = self.relu(self.conv64(x))    

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc1(x)
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

        self.fc1 = nn.Linear(size_fc_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
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


