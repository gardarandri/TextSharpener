


from PIL import Image, ImageDraw



im = Image.new("RGB", (1500,400), (255,255,255))
imd = ImageDraw.Draw(im)

def add_layer(num_channels, x, y, d, w, s):
    ox = x
    oy = y
    for i in range(num_channels):
        x = x + 0.7*d
        y = y + d
        imd.rectangle((x,y,x+w,y+w), outline=(0,0,0), fill=(170,170,170))

    imd.text((x,y+w+0.5*w),"{} channels".format(num_channels), fill=(0,0,0))
    imd.text((ox+num_channels*d/2+1.1*w,oy+num_channels*d/2),"{}".format(s), fill=(0,0,0))


nc = [3,16,32,32,32,32,32,32,16,3]
atx = 10
wd = [70] + [50] + [25]*6 + [50] + [70]
dif = [20] + [10] + [5]*6 + [10] + [20]
aty = [90]+[40]*9
offset = [50] + [0]*9
apstr = ["l_relu"]*8+["relu"]+[""]
for i in range(len(nc)):
    add_layer(nc[i], atx, aty[i], dif[i], wd[i], apstr[i])
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

im.save("architecture.png")







