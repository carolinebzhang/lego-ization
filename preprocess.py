import skimage
import numpy as np
from skimage import img_as_ubyte

def preprocess_image(image_path, target_size):
    image = skimage.io.imread(image_path)

    if image.shape[0] != image.shape[1]:
        diff = abs(image.shape[0] - image.shape[1])
        padding = [(0, 0), (0, 0), (0, 0)]
        if image.shape[0] > image.shape[1]:
            padding[1] = (diff // 2, diff - (diff // 2))
        else:
            padding[0] = (diff // 2, diff - (diff // 2))
        image = np.pad(image, padding, mode='constant', constant_values=255)

    image = skimage.transform.resize(image, (target_size, target_size))
    
    return image


