from models import CycleGAN
import torch
from datasets import get_data
import numpy as np
from utils import *
from PIL import Image as im

def generate_image(input_img_path, genre, model_path, output_image_path):

    model = CycleGAN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    input_image = process_single_image(input_img_path, 256)
    output_image = np.reshape(input_image, (256, 256, 3))

    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    data = im.fromarray(output_image)
    if data.mode != 'RGB':
        data = data.convert('RGB')
      
    # saving the final output 
    # as a PNG file
    final_path = "saving.png"
    data.save(final_path)
    #input_image = torch.permute(torch.tensor(input_image).double(),(2,0,1))
    input_image = torch.tensor(input_image)
    input_image = torch.reshape(input_image, (3, 256, 256))
    input_image = torch.unsqueeze(input_image, 0)
    model.mode = "A2B"
    
    # Generate a time-shift representation of the output song
    if genre == 'lego':
        output_image = model.G_A2B(input_image)
    elif genre == 'real_world':
        output_image = model.G_B2A(input_image)
    else:
        raise ValueError("Invalid genre specified") 
        #output_image=input_image 
    print(output_image.shape) 

    output_image = output_image.detach().cpu().numpy() 
    output_image = np.reshape(output_image, (256, 256, 3))

    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    
    data = im.fromarray(output_image)
    if data.mode != 'RGB':
        data = data.convert('RGB')
      
    # saving the final output 
    # as a PNG file
    final_path = output_image_path + ".png"
    data.save(final_path)

if __name__=="__main__":
    generate_image('HOUSE.jpg', genre='lego', model_path='model0.pth', output_image_path="bigmode4house.jpg")
    #generate_image('ORIGINAL.jpg', genre='lego', model_path='bad_discriminator.pth', output_image_path="bad_discriminator.jpg")