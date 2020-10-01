from PIL import Image
import numpy as np

im = np.array(Image.open("./output/test_0.png"))

print(im.shape)
print(im)
