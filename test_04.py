from PIL import Image, ImageDraw, ImageFont

def generate_bbox(char, font):
    character = char
    font_size = 48
    font_file = open(font, 'rb')
    font = ImageFont.truetype(font_file, font_size)
    font_width, font_height = font.getsize(character)
    return (0,0,font_width,font_height)

def generate(letter, prefix, suffix, font):
    word = prefix + letter + suffix
    back_color = 255
    image_size = 64
    font_size = 48
    font_file = open(font, 'rb')
    char_image = Image.new('L', (image_size, image_size), back_color)
    draw = ImageDraw.Draw(char_image)
    font = ImageFont.truetype(font_file, font_size)
    font_width, font_height = font.getsize(letter)
    pre_w, pre_h = font.getsize(prefix)
    x = (image_size - font_width)/2
    y = (image_size - font_height)/2
    draw.text((x-pre_w,y), word, 0, font=font)
    return char_image

print(generate_bbox('m', 'fonts/arial.ttf'))
print(generate_bbox('i', 'fonts/arial.ttf'))
print(generate_bbox('g', 'fonts/arial.ttf'))
prefix = 'ik'
suffix = 'lr'
letter = 'a'
generate(letter, prefix, suffix, 'fonts/arial.ttf').show()

