import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from create_lists import *
import random
from train_00 import Net, fit

class SpindleSet(Dataset):
    def __init__(self, char_list, font_list, length):
        self.char_list = char_list
        self.font_list = font_list
        self.length = length
    def generate(self, string, font):
        back_color = 255
        image_size = 64*len(string)
        font_size = 48
        font_file = open(font, 'rb')
        str_image = Image.new('L', (image_size, 64), back_color)
        draw = ImageDraw.Draw(str_image)
        font = ImageFont.truetype(font_file, font_size)
        font_width, font_height = font.getsize(string)
        x = (image_size - font_width)//2
        y = (image_size - font_height)//2
        offset_x = random.randint(-x,image_size-x-font_width) 
        offset_y = random.randint(-y,64-y-font_height)
        draw.text((x+offset_x,y+offset_y), string, 0, font=font)
        #str_image.show()
        return str_image
    def __len__(self):
        return len(self.char_list)**self.length*len(self.font_list)
    def __getitem__(self, idx):
        str_i = idx%(len(self.char_list)**self.length)
        font_i = idx//(len(self.char_list)**self.length)
        string = ""
        i = str_i
        for j in range(self.length):
            string += self.char_list[i%len(self.char_list)]
            i //= len(self.char_list)
        output = self.generate(string, self.font_list[font_i])
        return output, str_i

character_list = create_character_list()
#print(len(character_list))
font_list = ['fonts/Arial.ttf', 'fonts/Bodoni 72.ttc']
#font_list = create_font_list()
dataset = SpindleSet(character_list, font_list, 2)
#dataset[65+66*95]

class CarveSet(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self):
        width, height = self.img.size
        return (width-64+1)*(height-64+1)
    def __getitem__(self, idx):
        width, height = self.img.size
        i = idx//(width-64+1)
        j = idx%(height-64+1)
        x = self.img.crop((i, j, i+64, j+64))
        #x.show()
        return torch.from_numpy(np.array(x)).unsqueeze(0)/255

class WrapperSet(Dataset):
    def __init__(self, spindle_set):
        self.spindle_set = spindle_set
        self.model = Net()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
    def __len__(self):
        return len(self.spindle_set)
    def __getitem__(self, idx):
        spindle = self.spindle_set[idx]
        dataset = CarveSet(spindle[0])
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for xb in dataloader:
            yb = self.model(xb)
            #print(yb.permute((1,0)).unsqueeze(2).shape)
            return yb.permute((1,0)).unsqueeze(2), spindle[1]              

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv_layer = nn.Conv2d(95, 256, (5,1))
        #self.fc_layer = nn.Linear(256*61, 95*95)
        self.fc_layer = nn.Linear(65*95, 95*95)
    def forward(self, x):
        #x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.fc_layer(x)
        
dataset = WrapperSet(dataset)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
model = Net2()
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
fit(1000, model, loss_fn, opt, dataloader, dataloader)
