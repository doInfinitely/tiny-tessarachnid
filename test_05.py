from PIL import Image, ImageFont, ImageDraw

from train_00 import Net
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
    output = Image.new("L", ( w+64, 128), 255)
    draw = ImageDraw.Draw(output)
    draw.text((32,32), word, 0, font=font)
    return output

img1 = draw_word("g", 'fonts/Bodoni 72.ttc')
character_list = create_character_list()
def get_character_vector(img):
    model = Net(character_list)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    dataset = CarveSet(img)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    offset = 32
    lookup = dict()
    for xb, zb in dataloader:
        yb = model(xb)
        for j,y in enumerate(yb):
            rect = dataset.get_rect(zb[j])
            scores = {chr(k+offset):float(y[k]) for k in range(len(character_list))}
            lookup[rect] = scores
    return np.array([sorted([lookup[key][char] for key in lookup])[-1] for char in character_list])
img2 = draw_word("g", 'fonts/Arial.ttf')
lookup = dict()
'''
for x in ['baboon','giraffe','elephant', 'birth', 'brth', '', 'irthb']:
    img = draw_word(x, 'fonts/Arial.ttf')
    lookup[x] = get_character_vector(img)

img = draw_word("birth", 'fonts/Bodoni 72.ttc')
for x in lookup:
    print(x, np.linalg.norm(lookup[x]-get_character_vector(img)))
for x in ["the qui fox", "fox qui the", "dog"]:
    img = draw_word(x, 'fonts/Arial.ttf')
    lookup[x] = get_character_vector(img)
img = draw_word("the qui fox", 'fonts/Bodoni 72.ttc')
for x in lookup:
    print(x, np.linalg.norm(lookup[x]-get_character_vector(img)))
'''
#sys.exit()
to_scan = draw_word("the qui fox, the lazy dog", 'fonts/Arial.ttf')
to_scan.crop((88, 0, 280, 128)).show()
strings = [x+y+z for x in character_list for y in character_list for z in character_list]
short_list = ['a', 'b', '']
strings = {x+y+z for x in short_list for y in short_list for z in short_list}
strings = ['qui','','dog', 'fox', 'the', 'ox', ' fo', 'th', 't', 'x', ' fox ', 'the ', 'qui ', 'i fo', 'i fox', 'lazy', 'dog', ' lazy ']
def generate_table(strings):
    output = dict()
    for x in strings:
        img = draw_word(x, 'fonts/Arial.ttf')
        print(x)
        output[x] = get_character_vector(img)
    return output

def contains(container, containee):
    if container[0] <= containee[0] and container[1] <= containee[1]:
        if container[2] >= containee[2] and container[3] >= containee[3]:
            return True
    return False

def interval_overlap(int1, int2):
    if int1[1] < int2[0]:
        return False
    if int2[1] < int1[0]:
        return False
    return True
def overlap(rect1, rect2):
    int1 = (rect1[0], rect1[2])
    int2 = (rect2[0], rect2[2])
    int3 = (rect1[1], rect1[3])
    int4 = (rect2[1], rect2[3])
    return interval_overlap(int1, int2) and interval_overlap(int3, int4)

def rect_of_overlap(rect1, rect2):
    rect3 = (max(rect1[0],rect2[0]),max(rect1[1],rect2[1]),min(rect1[2],rect2[2]),min(rect1[3],rect2[3]))
    return rect3

def area(rect):
    return (rect[3]-rect[1])*(rect[2]-rect[0])

def overlap_set(s, rect):
    return sum(overlap(r,rect) for r in s)

def overlap_set_word(s, rect_word):
    rect = rect_word[0]
    word = rect_word[1]
    summa = 0
    for pair in s:
        if pair[1] == word:
            if overlap(rect, pair[0]):
                summa += 1
    return summa

def table_search(char_vector, table):
    mini = (None,None)
    for key in table:
        dist = np.linalg.norm(char_vector-table[key])
        if mini[0] is None or dist < mini[0]:
            mini = (dist, key)
    return mini
table = generate_table(strings)


def is_substring_and_overlap(grabbed1, grabbed2):
    grabbed = sorted([grabbed1, grabbed2])
    if grabbed[0][1] in grabbed[1][1] or grabbed[1][1] in grabbed[0][1]:
        if overlap(grabbed[0][0], grabbed[1][0]):
            return True
    return False

def scan(img, table):
    model = Net(character_list)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    dataset = CarveSet(img)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    offset = 32
    lookup = dict()
    for xb, zb in dataloader:
        yb = model(xb)
        for j,y in enumerate(yb):
            rect = dataset.get_rect(zb[j])
            scores = {chr(k+offset):float(y[k]) for k in range(len(character_list))}
            lookup[rect] = scores
    width, height = to_scan.size
    lookup2 = dict()
    lookup3 = dict()
    for i in range(width-64):
        for j in [0]:#range(height-64):
            for length in [192]:#range(64, 64*3):
                sublook = dict()
                search_rect = (i,j,i+length,j+128)
                if i+length > width:
                    continue
                for key in lookup:
                    if contains(search_rect, key):
                        sublook[key] = lookup[key]
                lookup2[search_rect] = np.array([sorted([sublook[key][char] for key in sublook])[-1] for char in character_list])
                lookup3[search_rect] = table_search(lookup2[search_rect], table)
                print(lookup3[search_rect])
    
    grabbed = set()
    temp = sorted([(-1*lookup3[key][0],(-1*key[0],-1*key[1]), key, lookup3[key]) for key in lookup3 if overlap_set_word(grabbed, (key, lookup3[key][1])) == 0])
    while len(temp):
        print(temp)
        grabbed.add((temp[-1][2], temp[-1][3][1]))
        print(temp[-1])
        temp = sorted([(-1*lookup3[key][0],(-1*key[0],-1*key[1]), key, lookup3[key]) for key in lookup3 if overlap_set_word(grabbed, (key, lookup3[key][1])) == 0])
    sorted_grabbed = sorted(grabbed)
    i = 0
    while i < len(sorted_grabbed):
        j = i + 1
        while j < len(sorted_grabbed):
            if is_substring_and_overlap(sorted_grabbed[i], sorted_grabbed[j]):
                if len(sorted_grabbed[i][1]) > len(sorted_grabbed[j][1]):
                    del sorted_grabbed[j]
                    j -= 1
                else:
                    del sorted_grabbed[i]
                    i -= 1
                    break
            j += 1
        i += 1
    print(sorted_grabbed)

scan(to_scan, table)



#pickle.dump(lookup,open('lookup.p', 'wb'))


# determines if a given rectangle is within another
# format (left, top, right, bottom)

def search_for_next(letter_rect, next_letter, lookup):
    letter_set = set()
    search_box = (letter_rect[0],letter_rect[1]-64,letter_rect[2]+64,letter_rect[3]+64)
    for key in lookup:
        if contains(search_box, key):
            print(lookup[key][next_letter])
    return letter_lookup3[key]

def score_pair(letter1, letter2, lookup):
    scores = dict()
    letter_lookup = sorted([(lookup[rect][letter1],rect) for rect in lookup], reverse=True)
    maxi = (None, None)
    for i,x in enumerate(letteir_lookup):
        letter_rect = letter_lookup[i][1]
        search_box = (letter_rect[0],letter_rect[1],letter_rect[2],letter_rect[3]+64)
        for key in lookup:
            if contains(search_box, key):
                score = lookup[letter_rect][letter1]*lookup[key][letter2]
                scores[(letter_rect, key)] = score
                if maxi[0] is None or score > maxi[0]:
                    maxi = (score, (letter_rect, key))
                    print(maxi)
    img.crop(maxi[1][0]).show()
    img.crop(maxi[1][1]).show()
    return scores

sys.exit()
ph_scores = score_pair('p','h',lookup)
ha_scores = score_pair('h','a',lookup)
maxi = (None, None)
for key1 in ph_scores:
    for key2 in ha_scores:
        if key1[1] == key2[0]:
            score = ph_scores[key1]*ha_scores[key2]
            if maxi[0] is None or score > maxi[0]:
                maxi = (score, (key1,key2))
img.crop(maxi[1][0][0]).show()
img.crop(maxi[1][0][1]).show()
img.crop(maxi[1][1][0]).show()
def score_all_pairs(lookup):
    scores = dict()
    for rect in lookup:
        search_box = (rect[0],rect[1],rect[2],rect[3]+64)
        for key in lookup:
            if contains(search_box, key):
                score = {letter1+letter2:lookup[rect][letter1]*lookup[key][letter2] for letter1 in character_list for letter2 in character_list}
                scores[(rect, key)] =score
    ph_lookup = sorted([(scores[key]['ph'],key) for key in scores], reverse=True)
    img.crop(ph_lookup[0][1][0]).show()
    img.crop(ph_lookup[0][1][1]).show()
score_all_pairs(lookup)
sys.exit()
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

'''
for rect1 in h_set:
    for rect2 in h_set:
        print(rect1, rect2, area(rect_of_overlap(rect1,rect2)))
'''
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

box_sequences = search_for_word_2("phant", lookup)
print('box sequences', box_sequences)
print(search_for_word_2("phaze", lookup))
for x in box_sequences:
    for y in x:
        img.crop(y).show()
    break
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
'''
letter = 'e'
for box in reverse_lookup[letter]:
    img.crop(box).show()
'''
#by_max = sorted(by_max, reverse=True)
#img.crop(by_max[0][2]).show()
#print(by_max)
