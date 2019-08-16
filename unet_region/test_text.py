from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


header = Image.fromarray(np.ones((256, 256*4)))
drawer = ImageDraw.Draw(header)
font = ImageFont.truetype("sans-serif.ttf", 70)

text_ = 'Hello\nWorld'

drawer.text((0, 0), text_, 0, font=font)

header = np.array(header)
plt.imshow(header, cmap='gray')
plt.show()

