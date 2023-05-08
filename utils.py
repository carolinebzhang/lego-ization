import skimage
import numpy as np
from skimage import img_as_ubyte
import os
import torch 
import PIL
from PIL import Image

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
            try:
                img_npy = np.load(os.path.join(folder_path, filename))
                img_tensor = torch.tensor(img_npy)
                img_tensor = torch.reshape(img_tensor, (3, 256, 256))
                img_npys.append(img_tensor)
            except Exception as e:
                continue
            #print(timeshift)
            
    #final_images = [torch.reshape(torch.tensor(img), (3, 256, 256)) for img in img_npys]

    print("loaded!") 
    return img_npys

def crop_center(image_path, array_path):
    """
    Crop the image to 256x256 in the center.
    """
    #image = Image.open("realworld_scenes/9104.jpg")
    image = Image.open(image_path)
    width, height = image.size
    if width>=256 and height>=256:
        left = (width - 256) / 2
        top = (height - 256) / 2
        right = (width + 256) / 2
        bottom = (height + 256) / 2
        image.crop((left, top, right, bottom))
        image = np.asarray(image)
        np.save(array_path, image, allow_pickle=True)

def preprocess_flickr_scenes(folder_path: str):
    subinterval=0
    new_path = 'flickr_scene_arrays/'
    obj_label = "flickr_scene"
    error=0
    for dirpath, dirs, images in os.walk(folder_path):
        for image_path in images:
            try: 
                file_name = f"{obj_label}_{subinterval}.npy"
                final_path = os.path.join(new_path, file_name)
                image_path = os.path.join(folder_path, image_path)
                crop_center(image_path, final_path)
                subinterval+=1
                #print("saved!")
            except Exception as e: 
                #print(e)
                error+=1
                continue
    print(f"finished saving {obj_label}, {error}")


def preprocess_lego_scene(folder_path: str, target_size:int):
    subinterval=0
    new_path = "lego_scene_arrays/"
    obj_label = "lego_scene"
    error = 0 
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
            except Exception as e: 
                error+=1
                continue
    print(f"finished saving {obj_label}, {error}")
def get_all_object_data():
    preprocess_image("/Users/carolinezhang/Downloads/lego-ization/lego_objects", 256, True)
    print("legos are done!")
    preprocess_image("/Users/carolinezhang/Downloads/lego-ization/realworld_objects", 256, False)
    print("real world objects are done!")

def get_all_scene_data():
    preprocess_flickr_scenes("realworld_scenes")
    preprocess_lego_scene("Lego Scenes", 256)

if __name__=="__main__":
    get_all_scene_data()
