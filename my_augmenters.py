from imgaug import augmenters as iaa
import numpy as np

def rescale_images(images, random_state, parents, hooks):

    result = []
    for image in images:
        image_aug = np.copy(image)
        if(image.dtype == np.uint8):
            image_aug = image_aug / 255
        result.append(image_aug)
    return result

void_fun = lambda x, random_state, parents, hooks : x

rescale_augmenter = iaa.Lambda(
    func_images=rescale_images,
    func_heatmaps=void_fun,
    func_keypoints=void_fun
)
