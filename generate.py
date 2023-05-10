from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
from utils import *
from PIL import Image as im

# load a saved model and use it to generate new translated images
def generate_image(input_img_path, genre, model_path, output_image_path):

    # load model
    model = CycleGAN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # preprocess image
    input_image = process_single_image(input_img_path, 256)
    output_image = np.reshape(input_image, (256, 256, 3))

    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    data = im.fromarray(output_image)

    if data.mode != 'RGB':
        data = data.convert('RGB')
      
    input_image = torch.tensor(input_image)
    input_image = torch.reshape(input_image, (3, 256, 256))
    input_image = torch.unsqueeze(input_image, 0)
    
    # domain A is lego, B is real world 
    if genre == 'lego': # going from lego style
        model.mode = "A2B"
        output_image = model.G_A2B(input_image.float())
    elif genre == 'real_world': # going from real world style
        model.mode = "B2A"
        output_image = model.G_B2A(input_image.float())
    else:
        raise ValueError("Invalid genre specified") 
    
    # post process and save
    output_image = output_image.detach().cpu().numpy() 
    output_image = np.reshape(output_image, (256, 256, 3))

    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    
    data = im.fromarray(output_image)
    if data.mode != 'RGB':
        data = data.convert('RGB')

    # saving the final output as a PNG file
    final_path = output_image_path + ".png"
    data.save(final_path)

if __name__=="__main__":
    generate_image('Lego input.jpeg', genre='real_world', model_path='scene_6.pth', output_image_path="lego_real_6")