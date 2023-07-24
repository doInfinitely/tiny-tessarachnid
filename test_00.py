# randomly place letters on an image
from PIL import Image, ImageDraw, ImageFont
import random
from torch.utils.data import Dataset, DataLoader
from train_00 import Net
import torch
import numpy as np

width = 512*4
height = 512*4
char_list = ["A"]

def generate(char, font):
    character = char
    back_color = 255
    image_size = 64
    font_size = 48
    font_file = open(font, 'rb')
    char_image = Image.new('L', (image_size, image_size), back_color)
    draw = ImageDraw.Draw(char_image)
    font = ImageFont.truetype(font_file, font_size)
    font_width, font_height = font.getsize(character)
    x = (image_size - font_width)/2
    y = (image_size - font_height)/2
    draw.text((x,y), character, 0, font=font)
    return char_image

def add_char_image(im, char_im):
    width, height = im.size
    x = random.randrange(width-64)
    y = random.randrange(height-64)
    print(x,y)
    for i in range(64):
        for j in range(64):
            im.putpixel((x+i,y+j),min(im.getpixel((x+i,y+j)), char_im.getpixel((i,j))))
    return im, (x,y)

def crop_image(im, coord):
    return im.crop((coord[0], coord[1], coord[0]+64, coord[1]+64))

class CropSet(Dataset):
    def __init__(self, crops):
        self.crops = crops
    def __len__(self):
        return len(self.crops)
    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.crops[idx][0])).unsqueeze(0)/255, self.crops[idx][1]

if __name__ == "__main__":
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
    crop_image(output, char_locations[0][1]).show()

    crops = [(crop_image(output, char_locations[i][1]), char_locations[i][0]) for i in range(len(char_locations))]

    dataset = CropSet(crops)
    dataloader = DataLoader(dataset, batch_size=len(crops), shuffle=False)
    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    for xb, yb in dataloader:
        zb = model(xb)
        for j,y in enumerate(yb):
            print(y, chr(torch.argmax(zb[j])+offset))
