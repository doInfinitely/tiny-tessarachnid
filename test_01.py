import torch
import numpy as np
from train import Net, SpiderSet
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import sys
from create_lists import *
import pickle
from test_00 import generate, add_char_image

width = 512*2
height = 512*2
class CarveSet(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self):
        width, height = self.img.size
        return (width-64)*(height-64)
    def __getitem__(self, idx):
        width, height = self.img.size
        i = idx//(height-64)
        j = idx%(height-64)
        x = self.img.crop((i, j, i+64, j+64))
        return torch.from_numpy(np.array(x)).unsqueeze(0)/255
if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    output = Image.new('L', (width, height), 255)
    char_locations = []
    for c in range(32,127):
        c = chr(c)
        char_im = generate(c, 'fonts/Arial.ttf')
        output, coord = add_char_image(output, char_im)
        char_locations.append((c,coord))
    print(char_locations)
    output.show()

    dataset = CarveSet(output)
    #Image.fromarray((dataset[10000]*255).squeeze(0).numpy()).show()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    t = pickle.load(open('thresholds.p','rb'))
    count = 0
    for xb in dataloader:
        M = [None, None, None]
        yb = model(xb)
        for j,y in enumerate(yb):
            val = y[72]
            chars = []
            for key in t:
                if t[key]*.5 <= y[ord(key)-32]:
                    #if count >= 100:
                    #    sys.exit(0) 
                    chars.append(key)
            if len(chars):
                if count % 1000 == 0:
                    Image.fromarray((xb[j]*255).squeeze(0).numpy()).show()
                    print(chars)
            count += 1
