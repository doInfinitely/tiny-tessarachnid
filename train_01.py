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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        offset_x = 0#random.randint(-x,image_size-x-font_width) 
        offset_y = 0#random.randint(-y,64-y-font_height)
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
character_list = ['a', 'b', 'c', 'd','e','f','g','h','i','j']
#character_list = ['f', 'i']
#print(len(character_list))
font_list = ['fonts/Arial.ttf', 'fonts/Bodoni 72.ttc']
#font_list = ['fonts/Arial.ttf']*64 + ['fonts/Bodoni 72.ttc']*64
#font_list = create_font_list()
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
        return (torch.from_numpy(np.array(x)).unsqueeze(0)/255).to(device)

class WrapperSet(Dataset):
    def __init__(self, spindle_set):
        self.spindle_set = spindle_set
        self.model = Net()
        self.model.to(device)
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
            return yb.permute((1,0)).unsqueeze(2).to(device), torch.from_numpy(np.array(spindle[1])).to(device)         

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(10, 256, (5,1))
        self.fc_layer1 = nn.Linear(256*61, 256*32)
        self.fc_layer2 = nn.Linear(256*32, 256*8)
        self.fc_layer3 = nn.Linear(256*8, 10*10)
        self.fc_layer = nn.Linear(65*10, 100)
        self.fc_layer2 = nn.Linear(10, 100)
        self.pool = nn.MaxPool2d((65,1))
    def forward(self, x):
        '''
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc_layer1(x)
        x = F.tanh(x)
        x = self.fc_layer2(x)
        x = F.tanh(x)
        x = self.fc_layer3(x)
        x = F.tanh(x)
        '''
        x = self.pool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        #x = self.fc_layer(x)
        #x = F.tanh(x)
        x = self.fc_layer2(x)
        return x

class WrapperSet2(Dataset):
    def __init__(self, spindle_set):
        self.spindle_set = spindle_set
    def __len__(self):
        return len(self.spindle_set)
    def __getitem__(self, idx):
        spindle = self.spindle_set[idx]
        return (torch.from_numpy(np.array(spindle[0])).unsqueeze(0)/255).to(device), torch.from_numpy(np.array(spindle[1])).to(device)         

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodel = Net()
        self.submodel.to(device)
        self.submodel.load_state_dict(torch.load("model.pth"))
        self.submodel.eval()
        #self.fc_layer1 = nn.Linear(288, 100)
        #self.fc_layer1 = nn.Linear(107520, 100)
        #self.fc_layer1 = nn.Linear(119040, 4)
        self.fc_layer1 = nn.Linear(430080, 1024)
        #self.fc_layer1 = nn.Linear(32, 1024)
        self.fc_layer2 = nn.Linear(1024, 100)
        self.conv_layer_sizes = [1, 256]
        self.convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        conv_size = 8
        for i in range(len(self.conv_layer_sizes)-1):
            self.convs.append(nn.Conv2d(self.conv_layer_sizes[i], self.conv_layer_sizes[i+1], conv_size))
    def forward(self, x):
        #print(x.shape)
        for y in self.convs:
            x = y(x)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.tanh(x)
        x = self.fc_layer1(x)
        x = F.tanh(x) 
        x = self.fc_layer2(x)
        return x
        
dataset = SpindleSet(character_list, font_list, 2)
dataset = WrapperSet2(dataset)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
loss_fn = nn.CrossEntropyLoss()
model = Net3().to(device)
opt = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
fit(1000, model, loss_fn, opt, dataloader, dataloader)

'''
dataset = WrapperSet(dataset)
dataloader = DataLoader(dataset, batch_size=128*16, shuffle=True)
model = Net2().to(device)
#temp = dataset[10]
#print(temp)
#print(model(temp[0]).shape)
#sys.exit()
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
fit(300, model, loss_fn, opt, dataloader, dataloader)
torch.save(model.state_dict(), "model_2.pth")
'''
for xb, yb, in dataloader:
    zb = model(xb)
    #print(zb.shape)
    for j,y in enumerate(yb):
        print(torch.argmax(zb[j]), y, zb[j][torch.argmax(zb[j])], zb[j][y])
        print(zb[j])

