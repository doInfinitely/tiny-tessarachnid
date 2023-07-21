import torch
from train import Net, SpiderSet, SpiderSet2
from torch.utils.data import Dataset, DataLoader
from create_lists import *
import pickle

# dumb method for finding cutoff threshold for character detection
def threshold(model, dl):
    t = dict()
    for xb, yb in dl:
        model.eval()
        with torch.no_grad():
            zb = model(xb)
            for j,y in enumerate(yb):
                #print(chr(y+32), zb)
                if chr(y+32) in t:
                    t[chr(y+32)] = min(t[chr(y+32)], float(zb[j][y]))
                else:
                    t[chr(y+32)] = float(zb[j][y])
    print(t)
    pickle.dump(t, open('thresholds.p', 'wb'))
    
if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    character_list = create_character_list()
    font_list = ['fonts/Arial.ttf', 'fonts/Bodoni 72.ttc']
    augmentation_set = {None}
    dataset = SpiderSet2(character_list, font_list, augmentation_set)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    threshold(model, dataloader)
