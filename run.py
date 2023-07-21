import torch
import numpy as np
from train import Net, SpiderSet
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import sys
from create_lists import *
character_list = ['a', 'b', 'c', 'd','e','f','g','h','i','j']
character_list = create_character_list()
font_set = {'fonts/Arial.ttf'}
augmentation_set = {None}
dataset = SpiderSet(character_list, font_set, augmentation_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()


Image.fromarray((dataset.data[1]*255).squeeze(0).numpy()).show()
for xb, yb in dataloader:
    print(model(xb))
    print(yb)
    break

class CarveSet(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self):
        width, height = self.img.size
        return (width-64)*(height-64)
    def __getitem__(self, idx):
        width, height = self.img.size
        i = idx//(width-64)
        j = idx%(height-64)
        x = self.img.crop((i, j, i+64, j+64))
        return torch.from_numpy(np.array(x)).unsqueeze(0)/255
im = Image.open("screenshot.png")
width, height = im.size
dataset = CarveSet(ImageOps.grayscale(im).resize((width*2,height*2)))
#Image.fromarray((dataset[10000]*255).squeeze(0).numpy()).show()
dataloader = DataLoader(dataset, batch_size=128*64, shuffle=False)
'''
for xb in dataloader:
    M = [None, None, None]
    for j,y in enumerate(yb):
        maxi = torch.argmax(y)
        val = y[maxi]
        if M[0] is None or val > y[maxi]:

'''
count = 0
for xb in dataloader:
    M = [None, None, None]
    yb = model(xb)
    for j,y in enumerate(yb):
        val = y[72]
        if M[0] is None or val > M[0]:
            M[0] = val
            M[1] = j
            M[2] = chr(torch.argmax(y)+32)
    count += 1
    Image.fromarray((xb[M[1]]*255).squeeze(0).numpy()).show()
    print(M[2])
'''
count = 0
for xb in dataloader:
    for j,y in enumerate(yb):
        if torch.argmax(y) == 70:
            count += 1
        if count == 10:
            Image.fromarray((xb[j]*255).squeeze(0).numpy()).show()
            print(y)
            sys.exit(0)
'''
