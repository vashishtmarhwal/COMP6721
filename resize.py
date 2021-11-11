from PIL import Image
import os, sys

path = "./FinalDataset/ffp2/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        print(path+item)
        if os.path.isfile(path+item):
            if "resized" not in path+item:
                im = Image.open(path+item)
                im = im.convert('RGB')
                f, e = os.path.splitext(path+item)
                imResize = im.resize((128,128), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=90)


resize()
for item in dirs:
    if os.path.isfile(path+item):
        if "resized" not in path+item:
            os.remove(path+item)