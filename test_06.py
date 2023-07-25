from create_lists import *
from test_05 import *
import PIL
from multiprocessing import Pool

def draw_word(word, font_path):
    font = ImageFont.truetype(font_path, 12)
    w, h = font.getsize(word)
    output = Image.new("L", ( w+64, 128), 255)
    draw = ImageDraw.Draw(output)
    draw.text((32,32), word, 0, font=font)
    return output

character_list = create_character_list()
class TextScanner():
    def __init__(self, img, table):
        self.img = img.convert('L')
        self.table = table
        #self.scales = [4]#,1,2,3]
        self.scales = [0.1,0.25,0.5,1,2,3,4]
        self.scales = [0.1,0.25,0.5,1,2,3]
        #self.scales = [0.25,0.5,1]
        self.word_scan_max_size = 4
        self.pyramid = dict()
        self.lookup = dict()
        self.word_lookup = dict()
        self.generate_pyramid()
    def generate_pyramid(self):
        width, height = self.img.size
        for x in self.scales:
            self.pyramid[x] = self.img.resize((int(width*x), int(height*x)))
    def level_scan(self, scale):
        model = Net(character_list)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        dataset = CarveSet(self.pyramid[scale])
        dataloader = DataLoader(dataset, batch_size=4096*8, shuffle=False)
        offset = 32
        output = dict()
        for xb, zb in dataloader:
            yb = model(xb)
            for j,y in enumerate(yb):
                rect = dataset.get_rect(zb[j])
                scores = {chr(k+offset):float(y[k]) for k in range(len(character_list))}
                print(scale, rect, scores)
                output[rect] = scores
        return output
    def table_search(self, character_vector):
        mini = (None,None)
        for key in self.table:
            dist = np.linalg.norm(character_vector-table[key])
            if mini[0] is None or dist < mini[0]:
                mini = (dist, key)
        return mini
    def generate_subrects(self, search_rect):
        return [(i, j, i+64, j+64) for i in range(search_rect[0],search_rect[2]-64) for j in range(search_rect[1], search_rect[3]-64)]
    def word_scan_helper(self, inpt):
        scale = inpt[0]
        search_rect = inpt[1]
        sublook = dict()
        for x in self.generate_subrects(search_rect):
            if x in self.lookup[scale]:
                sublook[x] = self.lookup[scale][x]
        character_vector = [0 for char in character_list]
        for key in sublook:
            for k in range(len(character_vector)):
                character_vector[k] = max(character_vector[k], sublook[key][chr(32+k)])
        character_vector = np.array(character_vector)
        return self.table_search(character_vector)

    def word_scan(self):
        queue = []
        for key in self.lookup:
            width, height = self.pyramid[key].size
            for i in range(0,width-64,max(1,key)):
                for j in range(0,height-64,max(1,key)):
                    for length in [64*self.word_scan_max_size]:
                        sublook = dict()
                        search_rect = (i,j,i+length,j+128)
                        queue.append((key, search_rect))
                        print(key, search_rect)
                        continue
                        #print(search_rect)
                        for x in self.generate_subrects(search_rect):
                            if x in self.lookup[key]:
                                sublook[x] = self.lookup[key][x]
                        '''
                        if search_rect[2] > width or search_rect[3] > height:
                            continue
                        for key2 in self.lookup[key]:
                            if contains(search_rect, key2):
                                sublook[key2] = self.lookup[key][key2]
                        '''
                        character_vector = [0 for char in character_list]
                        for key2 in sublook:
                            for k in range(len(character_vector)):
                                character_vector[k] = max(character_vector[k], sublook[key2][chr(32+k)])
                        character_vector = np.array(character_vector)
                        #character_vector = np.array([sorted([sublook[key][char] for key in sublook])[-1] for char in character_list])
                        self.word_lookup[key, search_rect] = self.table_search(character_vector)
                        print(key, search_rect, tuple(np.array(search_rect)/key), self.word_lookup[key, search_rect])
        with Pool(32) as p:
            output = p.map(self.word_scan_helper, queue)
        for k,z in enumerate(queue):
            self.word_lookup[z] = output[k]
    '''
    def divide(self, rect):
        output = []
        width = rect[2]-rect[0]
        height = rect[3]-rect[1]
        output.append((rect[0],rect[1],rect[0]+width//2,rect[1]+height//2))
        output.append((rect[0]+width//2,rect[1],rect[0]+width,rect[1]+height//2))
        output.append((rect[0],rect[1]+height//2,rect[0]+width//2,rect[1]+height))
        output.append((rect[0]+width//2,rect[1]+height//2,rect[0]+width,rect[1]+height))
        return output
    '''
    def subdivide(self, rect, scale=1):
        output = []
        width = rect[2]-rect[0]
        height = rect[3]-rect[1]
        if rect[0] < rect[0]+width-scale: 
            output.append((rect[0],rect[1],rect[0]+width-scale,rect[1]+height))
        if rect[1] < rect[1]+width-scale: 
            output.append((rect[0],rect[1],rect[0]+width,rect[1]+height-scale))
        if rect[0]+scale < rect[0]+width: 
            output.append((rect[0]+scale,rect[1],rect[0]+width,rect[1]+height))
        if rect[1]+scale < rect[1]+height: 
            output.append((rect[0],rect[1]+scale,rect[0]+width,rect[1]+height))
        return output
    def box_scan(self, scale, search_rect=None): 
        if search_rect is None:
            width, height = self.pyramid[scale].size
            search_rect = (0,0, width, height)
        sublook = dict()
        for key in self.lookup[scale]:
            if contains(search_rect, key):
                sublook[key] = self.lookup[scale][key]
        try:
            character_vector = np.array([sorted([sublook[key][char] for key in sublook])[-1] for char in character_list])
        except IndexError:
            return (None, None)
        self.word_lookup[scale, search_rect] = self.table_search(character_vector)
        print(search_rect, tuple(np.array(search_rect)/scale), self.word_lookup[scale, search_rect])
    def character_scan(self):
        for scale in self.scales:
            self.lookup[scale] = self.level_scan(scale)
    def scan(self):
        self.character_scan()
        self.word_scan()
        self.global_word_lookup = dict()
        for key in self.word_lookup:
            self.global_word_lookup[tuple(np.array(key[1])/key[0])] = self.word_lookup[key]
    def stitch(self):
        grabbed = set()
        lookup = self.global_word_lookup
        temp = sorted([(lookup[key][0],(key[1],key[0]), key, lookup[key]) for key in lookup if overlap_set_word(grabbed, (key, lookup[key][1])) == 0])
        while len(temp):
            grabbed.add((temp[0][2], temp[0][3][1]))
            print(temp[0])
            temp = sorted([(lookup[key][0],(key[1],key[0]), key, lookup[key]) for key in lookup if overlap_set_word(grabbed, (key, lookup[key][1])) == 0])
        grabbed = sorted(grabbed)
        i = 0
        while i < len(grabbed):
            j = i + 1
            while j < len(grabbed):
                if is_substring_and_overlap(grabbed[i], grabbed[j]):
                    if len(grabbed[i][1]) > len(grabbed[j][1]):
                        del grabbed[j]
                        j -= 1
                    else:
                        del grabbed[i]
                        i -= 1
                        break
                j += 1
            i += 1
        return grabbed
img = draw_word("THANK YOU VERY MUCH", 'fonts/Arial.ttf')
img = draw_word("MUCH", 'fonts/Arial.ttf')
im = Image.open("004-cropped2.jpg")
#im.show()
img.crop((16.0, 25.0, 101.33333333333333, 67.66666666666667)).show()
table = generate_table(['MUCH', 'THANK', 'YOU', 'YOU VERY', 'UCH','VERY',''])
scanner = TextScanner(img, table)
#scanner.pyramid[3].crop((0, 22, 192, 150)).show()
scanner.scan()
#sys.exit()
temp = scanner.stitch()
print(temp)
#print(scanner.global_word_lookup)
#img.crop(temp[0][0]).show()
