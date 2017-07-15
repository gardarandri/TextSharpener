


from PIL import Image, ImageDraw, ImageFont

text_font = ImageFont.truetype("arial.ttf", 20)


im = Image.new("RGB", (1400,400), (255,255,255))
imd = ImageDraw.Draw(im)

def add_layer(num_channels, x, y, d, w, s, s2, cv):
    ox = x
    oy = y
    for i in range(num_channels):
        x = x + 0.7*d
        y = y + d
        if cv == None:
            imd.rectangle((x,y,x+w,y+w), outline=(0,0,0), fill=(170,170,170))
        else:
            imd.rectangle((x,y,x+w,y+w), outline=(0,0,0), fill=cv[i % len(cv)])

    imd.text((x,y+w+0.5*w),"{} channels".format(num_channels), fill=(0,0,0), font=text_font)
    imd.text((ox+num_channels*d/2+w+d,oy+num_channels*d/2),"{}".format(s), fill=(0,0,0), font=text_font)
    imd.text((ox+num_channels*d/2+w+d,oy+num_channels*d/2 + 13),"{}".format(s2), fill=(0,0,0), font=text_font)


nc = [3,16,32,32,32,32,32,16,3]
atx = 10
wd = [70] + [50] + [25]*5 + [50] + [70]
dif = [20] + [10] + [5]*5 + [10] + [20]
aty = [90]+[40]*8
offset = [70] + [0]*6 + [30] + [0]
apstr = ["conv"]*4 + ["deconv"]*4 + [""]
apstr2 = ["l_relu"]*7+["relu"]+[""]
colvec = [[(255,0,0),(0,255,0),(0,0,255)]] + [None]*7 + [[(255,0,0),(0,255,0),(0,0,255)]]
for i in range(len(nc)):
    add_layer(nc[i], atx, aty[i], dif[i], wd[i], apstr[i], apstr2[i], colvec[i])
    atx += 0.7*dif[i]*nc[i] + wd[i] + offset[i]

#add_layer(ax, 3,  0.04, 0.4, 0.02, 0.1)
#add_layer(ax, 16, 0.04, 0.4, 0.008, 0.1)
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#add_layer(ax, 32, 0.04, 0.4, 0.004, 0.1)
#add_layer(ax, 16, 0.04, 0.4, 0.008, 0.1)
#add_layer(ax, 3,  0.04, 0.4, 0.02, 0.1)

im.save("test.png")







