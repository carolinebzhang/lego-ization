import skimage
import numpy as np
from skimage import img_as_ubyte
import os
import torch 

def preprocess_image(folder_path:str, target_size:int, is_lego:bool):
    
    subinterval=0
    if is_lego:
        new_path = "lego_arrays/"
        obj_label = "lego"
    else:
        new_path = "object_arrays/"
        obj_label = "object"
    for dirpath, dirs, images in os.walk(folder_path):
        for image_path in images:
            try: 
                image_path = os.path.join(folder_path, image_path)
                image = skimage.io.imread(image_path, as_gray=False)
                if image.shape[0] != image.shape[1]:
                    diff = abs(image.shape[0] - image.shape[1])
                    padding = [(0, 0), (0, 0), (0, 0)]
                    if image.shape[0] > image.shape[1]:
                        padding[1] = (diff // 2, diff - (diff // 2))
                    else:
                        padding[0] = (diff // 2, diff - (diff // 2))
                    image = np.pad(image, padding, mode='constant', constant_values=255)
                image = skimage.transform.resize(image, (target_size, target_size))
                file_name = f"{obj_label}_{subinterval}.npy"
                final_path = os.path.join(new_path, file_name)
                np.save(final_path, image, allow_pickle=True)
                subinterval+=1
                print("saved!")
            except Exception as e: 
                print(e)
                continue
    print("finished saving {obj_label}")

def process_single_image(image_path, target_size):
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

def numpy_to_torch(folder_path):
    """
    Loads numpy arrays in a folder and returns them as a list of numpy arrays.

    Args:
    - folder_path: string, the path to the folder containing the numpy arrays to be unpickled.

    Returns:
    - data: list of numpy arrays, contains all unpickled numpy arrays from the folder.
    """
    img_npys = []
    for filename in os.listdir(folder_path):
    # check if the file is a .npy file
        if filename.endswith(".npy"):
            # load the array from the file
            img_npy = np.load(os.path.join(folder_path, filename))
            img_npys.append(img_npy)
            #print(timeshift)
            
    final_images = [torch.reshape(torch.tensor(img), (3, 256, 256)) for img in img_npys]

    print("loaded!") 
    return final_images


def get_all_data():
    preprocess_image("/Users/carolinezhang/Downloads/lego-ization/lego_objects", 256, True)
    print("legos are done!")
    preprocess_image("/Users/carolinezhang/Downloads/lego-ization/realworld_objects", 256, False)
    print("real world objects are done!")

get_all_data()
