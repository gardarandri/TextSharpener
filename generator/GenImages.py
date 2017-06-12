
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random





im_font = ImageFont.truetype("./FreeMono.otf",size=30)

def random_coord(a,b):
    return (int(random.random()*a),int(random.random()*b))

def random_string():
    res = ""
    for _ in range(10):
        res = res + chr(60+int(random.random()*50))

    return res

def random_color():
    return ( int(random.random()*256), int(random.random()*256), int(random.random()*256))

def gen_image(save_path, dataset_name, number_of_images):
    for im_num in range(number_of_images):
        background_color = random_color()
        im_real = Image.new("RGB", (100,100),background_color)
        im_distorted = Image.new("RGB", (100,100),background_color)
    
        im_real_d = ImageDraw.Draw(im_real)
        im_distorted_d = ImageDraw.Draw(im_distorted)
    
        rand_pos = random_coord(10,50)
        rand_str = random_string()
    
        text_color = random_color()
        im_real_d.text(rand_pos, rand_str, font=im_font, fill=text_color)
        im_distorted_d.text(rand_pos, rand_str, font=im_font, fill=text_color)
    
        rand_rotation = 90*random.random()
        im_real = im_real.rotate(rand_rotation)
        im_distorted = im_distorted.rotate(rand_rotation)

        im_distorted = im_distorted.filter(ImageFilter.GaussianBlur(radius=1.8*random.random()+0.7))
    
        im_real.save(save_path+"/"+dataset_name+"_label"+str(im_num)+".png")
        im_distorted.save(save_path+"/"+dataset_name+"_train"+str(im_num)+".png")


gen_image("../data","set_1",1000)
gen_image("../data","validation_1",10)


