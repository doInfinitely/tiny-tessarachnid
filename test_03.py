from PIL import Image, ImageFont, ImageDraw

from train import Net
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from create_lists import *

class CarveSet(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self):
        width, height = self.img.size
        return (width-64+1)*(height-64+1)
    def __getitem__(self, idx):
        width, height = self.img.size
        i = idx//(height-64+1)
        j = idx%(height-64+1)
        x = self.img.crop((i, j, i+64, j+64))
        return torch.from_numpy(np.array(x)).unsqueeze(0)/255, idx
    def get_rect(self, idx):
        width, height = self.img.size
        i = int(idx)//(height-64+1)
        j = int(idx)%(height-64+1)
        return (i,j,i+64,j+64)

def draw_word(word, font_path):
    font = ImageFont.truetype(font_path, 48)
    w, h = font.getsize(word)
    output = Image.new("L", ( w+64*2, 96), 255)
    draw = ImageDraw.Draw(output)
    draw.text((64,32), word, 0, font=font)
    return output

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()
img = draw_word("hello", 'fonts/Arial.ttf')
img.show()
dataset = CarveSet(img)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
t = pickle.load(open('thresholds.p','rb'))

lookup = dict()
for xb, zb in dataloader:
    try:
        lookup = pickle.load(open('lookup.p','rb'))
        break
    except:
        pass
    yb = model(xb)
    for j,y in enumerate(yb):
        val = y[72]
        chars = []
        maxi = torch.argmax(y)
        if y[maxi] > t[chr(maxi+32)]*0.9:
            chars.append(chr(maxi+32))
        '''
        for key in t:
            if t[key]*0.9 <= y[ord(key)-32]:
            #if t[' ']*0.3 <= y[ord(key)-32]:
            #if 10 <= y[ord(key)-32]:
                chars.append(key)
        '''
        if len(chars):
            #Image.fromarray((xb[j]*255).squeeze(0).numpy()).show()
            rect = dataset.get_rect(zb[j])
            print(rect, chars)
            lookup[rect] = chars
pickle.dump(lookup,open('lookup.p', 'wb'))

print("finding h")
h_set = set()
for rect in lookup:
    if "h" in lookup[rect]:
        h_set.add(rect)

print(h_set)

# determines if a given rectangle is within another
# format (left, top, right, bottom)
def contains(container, containee):
    if container[0] <= containee[0] and container[1] <= containee[1]:
        if container[2] >= containee[2] and container[3] >= containee[3]:
            return True
    return False

# search for e next to h
for rect in h_set:
    print('h', rect)
    e_set = set()
    search_box = (rect[0],rect[1]-64,rect[2]+64,rect[3]+64)
    for key in lookup:
        if contains(search_box, key):
            if 'e' in lookup[key]:
                e_set.add(key)
    print('e', e_set)

def search_for_next(letter_rect, next_letter, lookup):
    letter_set = set()
    search_box = (letter_rect[0],letter_rect[1]-64,letter_rect[2]+64,letter_rect[3]+64)
    for key in lookup:
        if contains(search_box, key):
            if next_letter in lookup[key]:
                letter_set.add(key)
    return letter_set

def search_for_substring(letter_rect, substring, lookup):
    if not len(substring):
        return set([tuple(),])
    output = set()
    print(substring[0])
    letter_set = search_for_next(letter_rect, substring[0], lookup)
    for rect in letter_set:
        next_letters = search_for_substring(rect, substring[1:], lookup)
        print(next_letters)
        for x in next_letters:
            output.add((rect,)+x)
    return output

def search_for_word(word, lookup):
    first = True
    output = set()
    letter_set = set()
    for letter in word:
        if first:
            first = False
            for key in lookup:
                if letter in lookup[key]:
                    letter_set.add(key)
        else:
            for rect in letter_set:
                next_set = search_for_substring(rect, word[1:], lookup)
                for x in next_set:
                    output.add((rect,)+x)
    return output

# takes forever to run because of the branch factor
#print(search_for_word("hello", lookup))
print(search_for_word("he", lookup))
# first need to look for letter "islands"

def rect_of_overlap(rect1, rect2):
    rect3 = (max(rect1[0],rect2[0]),max(rect1[1],rect2[1]),min(rect1[2],rect2[2]),min(rect1[3],rect2[3]))
    return rect3

def area(rect):
    return (rect[3]-rect[1])*(rect[2]-rect[0])

for rect1 in h_set:
    for rect2 in h_set:
        print(rect1, rect2, area(rect_of_overlap(rect1,rect2)))

def connected_components(box_set, overlap_threshold):
    components = [{x,} for x in box_set]
    while True:
        quadbreak = False
        for i,x in enumerate(components):
            for j,y in enumerate(components):
                if i < j:
                    for z in x:
                        for w in y:
                            if area(rect_of_overlap(z,w))/min(area(z),area(w)) >= overlap_threshold:
                                x.update(y)
                                del components[j]
                                quadbreak = True
                                break
                        if quadbreak:
                            break
                if quadbreak:
                    break
            if quadbreak:
                break
        else:
            break
    return components

components = connected_components(h_set, .9)
print(len(components))

def overlap_score(box_set):
    score = dict()
    for x in box_set:
        for y in box_set:
            a = area(rect_of_overlap(x,y))
            if x not in score:
                score[x] = 0
            if y not in score:
                score[y] = 0
            score[x] += a
            score[y] += a
    return score

#scores = overlap_score(components[0])
#print(scores)
def get_max_scoring(scores):
    return sorted([(scores[key], key) for key in scores])[-1][1]

#print(get_max_scoring(scores))

def search_for_substring_2(letter_rect, substring, lookup):
    if not len(substring):
        return set([tuple(),])
    output = set()
    letter_set = search_for_next(letter_rect, substring[0], lookup)
    components = connected_components(letter_set, .9)
    for c in components:
        rect = get_max_scoring(overlap_score(c))
        next_letters = search_for_substring_2(rect, substring[1:], lookup)
        for x in next_letters:
            output.add((rect,)+x)
    return output

def search_for_word_2(word, lookup):
    first = True
    output = set()
    letter_set = set()
    for key in lookup:
        if word[0] in lookup[key]:
            letter_set.add(key)
    components = connected_components(letter_set, .9)
    for c in components:
        rect = get_max_scoring(overlap_score(c))
        next_set = search_for_substring_2(rect, word[1:], lookup)
        for x in next_set:
            output.add((rect,)+x)
    return output

box_sequences = search_for_word_2("hello", lookup)
print(box_sequences)
'''
for x in box_sequences:
    for y in x:
        img.crop(y).show()
    break
'''
#character_list = create_character_list
reverse_lookup = dict()
for key in lookup:
    for x in lookup[key]:
        if x not in reverse_lookup:
            reverse_lookup[x] = []
        reverse_lookup[x].append(key)

for key in reverse_lookup:
    print(key, len(reverse_lookup[key]))
'''
for key in ['r','h']:
    if len(key.strip()):
        #print(key, len(reverse_lookup[key]))
        letter_set = set(reverse_lookup[key])
        #search_box = (0,0,128,128)
        #letter_set = {x for x in letter_set if contains(search_box, x)}
        print(key, len(letter_set))
        components = connected_components(letter_set, .9)
        print(key, [len(x) for x in components])
'''
letter = 'o'
for box in reverse_lookup[letter]:
    img.crop(box).show()
