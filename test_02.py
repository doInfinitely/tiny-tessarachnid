import torch
import numpy as np
from train_00 import Net, SpiderSet
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import sys
from create_lists import *
import pickle
from test_00 import generate, add_char_image, crop_image, CropSet

width = 512*2
height = 512*2
model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()
class CarveSet(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self):
        width, height = self.img.size
        return (width-64+1)*(height-64+1)
    def __getitem__(self, idx):
        width, height = self.img.size
        i = idx//(width-64+1)
        j = idx%(width-64+1)
        x = self.img.crop((i, j, i+64, j+64))
        return torch.from_numpy(np.array(x)).unsqueeze(0)/255

output = Image.new('L', (width, height), 255)
char_locations = []
offset = 97
for c in range(offset,127):
    c = chr(c)
    char_im = generate(c, 'fonts/Arial.ttf')
    output, coord = add_char_image(output, char_im)
    char_locations.append((c,coord))
print(char_locations)
output.show()
crops = [(crop_image(output, char_locations[i][1]), char_locations[i][0]) for i in range(len(char_locations))]
dataset = CropSet(crops)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
t = pickle.load(open('thresholds.p','rb'))
count = 0
for xb, zb in dataloader:
    M = [None, None, None]
    yb = model(xb)
    for j,y in enumerate(yb):
        #val = y[72]
        chars = []
        maxi = (None, None)
        for key in t:
            if t[key]*.9 <= y[ord(key)-offset]:
                #if count >= 100:
                #    sys.exit(0) 
                chars.append(key)
            if maxi[0] is None or maxi[0] < y[ord(key)-offset]:
                maxi = (y[ord(key)-offset], key)
            #Image.fromarray((xb[j]*255).squeeze(0).numpy()).show()
        print(zb[j], chars, maxi, y[ord(zb[j])-offset] if zb[j] <= 'j' else None)
        count += 1
