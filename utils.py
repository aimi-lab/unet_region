import numpy as np
from bounding_box import BoundingBox
from itertools import product

def get_opt_box(truth, rotations=False):

    n_coords = 20
    n_angles = 16

    shape = truth.shape

    loc = [truth.shape[0] // 2, truth.shape[1] // 2]

    half_width = np.unique(
        np.linspace(5, truth.shape[0] // 2 - 1, n_coords, dtype=int))

    if rotations:
        angle = np.linspace(0, np.pi, n_angles)
    else:
        angle = [0.]

    dims = product(half_width, half_width, angle)
    boxes = []
    
    for hh, ww, a in dims:
        R = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
        top_left = np.dot(np.array((-ww, +hh)), R)
        bottom_right = np.dot(np.array((+ww, -hh)), R)

        i = sorted((int(top_left[1] + loc[0]), int(bottom_right[1] + loc[0])))
        j = sorted((int(top_left[0] + loc[1]), int(bottom_right[0] + loc[1])))
        boxes.append(BoundingBox(corners=(i, j),
                     orig_shape=truth.shape[:2]))

    boxes = [{
        'box': b,
        'mask': b.get_ellipse_mask(shape).astype(bool)
    } for b in boxes]

    boxes = [
        b for b in boxes
        if (not np.any(b['mask'] * np.logical_not(truth)))
    ]

    if(len(boxes) == 0):
        box = BoundingBox(
            corners=((loc[0] - 1, loc[0] + 1),
                        (loc[0] - 1, loc[0] + 1)),
            orig_shape=truth.shape[:2])
        boxes = [{'box': box,
                    'mask': box.get_ellipse_mask(shape).astype(bool)}]

    biggest = np.argmax([np.sum(b['mask']) for b in boxes])
    return boxes[biggest]
