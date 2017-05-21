
from PIL import Image, ImageDraw, ImageFont

import random





im_font = ImageFont.truetype("./FreeMono.otf",size=30)

def random_coord(a,b):
    return (int(random.random()*a),int(random.random()*b))

def random_string():
    res = ""
    for _ in range(10):
        res = res + chr(60+int(random.random()*50))

    return res

def gen_image(file_name):
    im = Image.new("RGB", (100,100))
    im_d = ImageDraw.Draw(im)
    im_d.text(random_coord(10,50), random_string(), font=im_font)
    im = im.rotate(90*random.random(), expand=1)
    im.save(file_name+".png")


for i in range(10):
    gen_image("./test"+str(i))


