#import augly.image as imaugs
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from create_lists import *
import random

character_list = ['a', 'b', 'c', 'd','e','f','g','h','i','j',]
character_list = create_character_list()
#print(len(character_list))
font_list = ['fonts/Arial.ttf', 'fonts/Bodoni 72.ttc']
#font_list = create_font_list()
augmentation_set = {None}

class SpiderSet(Dataset):
    def __init__(self, char_list, font_list, aug_set):
        self.data = []
        self.labels = []
        self.char_list = char_list
        self.spin_up(char_list, font_list, aug_set)
    def generate(self, char_i, font, aug):
        character = self.char_list[char_i]
        back_color = 255
        image_size = 64
        font_size = 48
        font_file = open(font, 'rb')
        char_image = Image.new('L', (image_size, image_size), back_color)
        draw = ImageDraw.Draw(char_image)
        #print(font_file)
        font = ImageFont.truetype(font_file, font_size)
        font_width, font_height = font.getsize(character)
        x = (image_size - font_width)/2
        y = (image_size - font_height)/2
        draw.text((x,y), character, 0, font=font)
        #char_image.show()
        if aug is not None:
            self.data.append(torch.from_numpy(np.array(aug(char_image))).unsqueeze(0))
        else:
            self.data.append(torch.from_numpy(np.array(char_image)).unsqueeze(0))
        self.data[-1] = self.data[-1]/255
        self.labels.append(char_i)
    def spin_up(self, char_list, font_list, aug_set):
        for font in font_list:
            for char_i in range(len(char_list)):
                for aug in aug_set:
                    self.generate(char_i, font, aug) 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SpiderSet2(Dataset):
    def __init__(self, char_list, font_list):
        self.char_list = char_list
        self.font_list = font_list
    def generate(self, character, prefix, suffix, font):
        word = prefix + character + suffix
        back_color = 255
        image_size = 64
        font_size = 48
        font_file = open(font, 'rb')
        char_image = Image.new('L', (image_size, image_size), back_color)
        draw = ImageDraw.Draw(char_image)
        #print(font_file)
        font = ImageFont.truetype(font_file, font_size)
        font_width, font_height = font.getsize(character)
        pre_w, pre_h = font.getsize(prefix)
        x = (image_size - font_width)//2
        y = (image_size - font_height)//2
        offset_x = 0#random.randint(-x,image_size-x-font_width) 
        offset_y = random.randint(-y,image_size-y-font_height) 
        draw.text((x-pre_w+offset_x,y+offset_y), word, 0, font=font)
        #char_image.show()
        return torch.from_numpy(np.array(char_image)).unsqueeze(0)/255
    def __len__(self):
        return len(self.char_list)*len(font_list)
    def __getitem__(self, idx):
        font_i = idx//len(self.char_list)
        char_i = idx%len(self.char_list)
        fixes = [x+y for x in self.char_list for y in self.char_list]
        prefix = random.choice(fixes)
        suffix = random.choice(fixes)
        output = self.generate(self.char_list[char_i], prefix, suffix, self.font_list[font_i])
        return output, char_i
        

'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_sizes = [1, 6, 16]
        self.convs = [] 
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_sizes = [1, 128]
        self.fc_layer_sizes = [1024, 512, 128, 95]
        conv_size = 32
        self.convs = nn.ModuleList()
        for i in range(len(self.conv_layer_sizes)-1):
            self.convs.append(nn.Conv2d(self.conv_layer_sizes[i], self.conv_layer_sizes[i+1], conv_size))
        #self.convs = [nn.Conv2d(self.conv_layer_sizes[i], self.conv_layer_sizes[i+1], 5) for i in range(len(self.conv_layer_sizes)-1)] 
        self.pool = nn.MaxPool2d(2, 2)
        input_size = 64
        for y in self.convs:
            input_size -= (conv_size-1)
            input_size //= 2
        self.fc_layer_sizes = [input_size*input_size*self.conv_layer_sizes[-1]] + self.fc_layer_sizes
        self.fcs = nn.ModuleList()
        for i in range(len(self.fc_layer_sizes)-1):
            self.fcs.append(nn.Linear(self.fc_layer_sizes[i], self.fc_layer_sizes[i+1]))
        #self.fcs = [nn.Linear(self.fc_layer_sizes[i], self.fc_layer_sizes[i+1]) for i in range(len(self.fc_layer_sizes)-1)]
    def forward(self, x):
        for y in self.convs:
            x = y(x)
            x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        for j,y in enumerate(self.fcs):
            x = y(x)
            if j == len(self.fcs)-1:
                break
            x = F.relu(x)
        return x
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None: # for training set
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl=None):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            print(loss_batch(model, loss_fn, xb, yb, opt))

        if valid_dl is None:
            continue

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in valid_dl]
            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


if __name__ == "__main__":
    dataset = SpiderSet2(character_list, font_list)
    #dataset = SpiderSet(character_list, font_list, augmentation_set)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = Net()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    fit(1000, model, loss_fn, opt, dataloader, dataloader)
    torch.save(model.state_dict(), "model.pth")
