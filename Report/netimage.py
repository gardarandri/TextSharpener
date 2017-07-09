


import matplotlib.pyplot as plt
import matplotlib.patches as patches




fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.axis("off")

def add_layer(inax, num_channels, x, y, d):
    for i in range(num_channels):
        inax.add_patch(
                patches.Rectangle(
                    (x + d*i,y-d*i),
                    0.1,
                    0.1
                    )
                )

add_layer(ax, 3, 0.1, 0.6, 0.02)
add_layer(ax, 32, 0.3, 0.6, 0.004)
add_layer(ax, 32, 0.5, 0.6, 0.004)
add_layer(ax, 32, 0.7, 0.6, 0.004)

fig.savefig("test.png", bbox_inches="tight")







