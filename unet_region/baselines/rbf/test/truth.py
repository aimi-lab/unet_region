from os.path import join as pjoin
from skimage import io, draw, segmentation, color, transform


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
