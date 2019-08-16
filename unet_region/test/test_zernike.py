from os.path import join as pjoin
from skimage import io, draw, segmentation, color, transform
import mahotas
import numpy as np
import matplotlib.pyplot as plt
 
class ZernikeMoments:
    def __init__(self, radius, crop=True, degree=20):
        self.radius = radius
        self.crop = crop
        self.degree = degree
 
    def describe(self, image):
		# return the Zernike moments for the image
        m, n = image.shape
        if(self.crop):
            y_where = np.argwhere(np.sum(image, axis=1))
            y_min, y_max = np.min(y_where), np.max(y_where)
            x_where = np.argwhere(np.sum(image, axis=0))
            x_min, x_max = np.min(x_where), np.max(x_where)
            image = image[y_min:y_max, x_min:x_max]
            image = transform.resize(image, (m, n))
        return mahotas.features.zernike_moments(image,
                                                self.radius,
                                                degree=self.degree)

root_dir = '/home/ubelix/data_mystique/medical-labeling'

out_size = 256

# img = io.imread(pjoin(root_dir, 'Dataset20/input-frames/frame_0030.png'))
# truth = io.imread(
#     pjoin(root_dir, 'Dataset20/ground_truth-frames/frame_0030.png'))
# p_x, p_y = 143, 132

img = io.imread(pjoin(root_dir, 'Dataset01/input-frames/frame_0150.png'))
truth = (io.imread(
    pjoin(root_dir, 'Dataset01/ground_truth-frames/frame_0150.png'))[..., 0] > 0).astype(float)
p_x, p_y = 190, 100

# img = io.imread(pjoin(root_dir, 'Dataset30/input-frames/frame_0075.png'))[..., :3]
# truth = (io.imread(
#     pjoin(root_dir, 'Dataset30/ground_truth-frames/frame_0075.png'))[..., 0] > 0).astype(float)
# p_x, p_y = 150, 110

img = transform.resize(img, (out_size, out_size))
img_gray = color.rgb2gray(img)
truth = transform.resize(truth, (out_size, out_size))
truth_contour = (segmentation.find_boundaries(truth > 0))

zm = ZernikeMoments(np.max(img.shape))

z = []
z.append(zm.describe(truth))
truth = np.roll(truth, -15, axis=1)
z.append(zm.describe(truth))
truth = truth[:, ::-1]
z.append(zm.describe(truth))
truth = np.roll(truth, -15, axis=0)
z.append(zm.describe(truth))
truth = transform.rotate(truth, 20)
z.append(zm.describe(truth))

plt.ion()
plt.imshow(truth)
plt.show()

fig, ax = plt.subplots(len(z), 1)
ax = ax.flatten()
for i, moments in enumerate(z):
    ax[i].stem(moments)
    ax[i].grid()

fig.show()
