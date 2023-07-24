import torch
from train_00 import Net, SpiderSet, SpiderSet2
from torch.utils.data import Dataset, DataLoader
from create_lists import *
import pickle

# dumb method for finding cutoff threshold for character detection
def threshold(model, dl):
    t = dict()
    offset = 32
    for xb, yb in dl:
        model.eval()
        with torch.no_grad():
            zb = model(xb)
            for j,y in enumerate(yb):
                #print(chr(y+32), zb)
                if chr(y+offset) in t:
                    t[chr(y+offset)] = min(t[chr(y+offset)], float(zb[j][y]))
                else:
                    t[chr(y+offset)] = float(zb[j][y])
    print(t)
    pickle.dump(t, open('thresholds.p', 'wb'))
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    character_list = create_character_list()
    #character_list = ['a', 'b', 'c', 'd','e','f','g','h','i','j']
    model = Net(character_list).to(device)
    model.load_state_dict(torch.load("model.pth"))
    font_list = ['fonts/Arial.ttf', 'fonts/Bodoni 72.ttc']
    dataset = SpiderSet2(character_list, font_list)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    threshold(model, dataloader)
