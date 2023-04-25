import matplotlib.pyplot as plt
import numpy as np

# create some random images to display in the subplots
images = [np.random.rand(50, 50) for i in range(4)]

# create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2)

# display the images in each subplot using imshow
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Image {i+1}")
    ax.set_xticks([])
    ax.set_yticks([])

# set the overall title for the plot
fig.suptitle("Four Random Images")

# display the plot
plt.show()
