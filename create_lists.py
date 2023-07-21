import os

def create_font_list():
    output = []
    exclude = {"NISC18030.ttf"}
    font_dir = '/System/Library/Fonts/Supplemental'
    for root, dirs, files in os.walk(font_dir):
        for f in files:
            if f not in exclude:
                output.append(os.path.join(root,f))
    return output

def create_character_list():
    return [chr(x) for x in range(32,127)]
