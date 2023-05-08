import numpy as np
 
import numpy as np
import torch as torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from utils import *

class ImageDataset(Dataset):
    def __init__(self, lego_objects, real_objects, lego_label, real_label):
        self.lego_imgs = lego_objects
        self.real_imgs = real_objects
        self.lego_label = lego_label 
        self.real_label = real_label

    def __len__(self):
        return len(self.lego_imgs)

    def __getitem__(self, index):
        lego_image = self.lego_imgs[index]
        real_image = self.real_imgs[index]
        lego_image = torch.squeeze(lego_image)
        real_image = torch.squeeze(real_image)
        get_index = np.array([index])
        sample = {'index': get_index, 'lego_image': lego_image, 'real_label': self.real_label,
                  'real_image': real_image, 'lego_label': self.lego_label}
        return sample
    
class ClassifierDataset(Dataset):
    def __init__(self, all_imgs, all_labels):
        self.all_imgs = all_imgs
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        img = self.all_imgs[index]
        img = torch.squeeze(img)
        img_label = self.all_labels[index]
        sample = {'img': img, 'img_classify': img_label}
        return sample

def get_data(batch_size):
    lego_samples = numpy_to_torch("lego_arrays")
    object_samples = numpy_to_torch("object_arrays")

    num_samples = min(len(lego_samples), len(object_samples))
    num_samples_round = num_samples // batch_size
    lego_samples = lego_samples[:num_samples_round*batch_size]
    object_samples = object_samples[:num_samples_round*batch_size]
    lego_real_set = ImageDataset(lego_objects=lego_samples, real_objects=object_samples, lego_label=[0,1], real_label=[1,0])

    training_set, testing_set = data.random_split(lego_real_set, [int(round(len(lego_real_set)*0.8)), int(round(len(lego_real_set)*0.2))])
    
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False)
    print('converted to dataloaders!')
    return training_loader, testing_loader

def get_classifier_data(batch_size):
    lego_samples = numpy_to_torch("lego_arrays")
    object_samples = numpy_to_torch("object_arrays")

    lego_num = len(lego_samples) // batch_size
    obj_num = len(object_samples) // batch_size

    lego_samples = lego_samples[:lego_num*batch_size]
    object_samples = object_samples[:obj_num*batch_size]

    lego_labels = [torch.nn.functional.one_hot(torch.tensor(1), num_classes=(2)).float() for i in lego_samples]
    object_labels = [torch.nn.functional.one_hot(torch.tensor(0), num_classes=(2)).float() for i in object_samples]

    all_samples = torch.cat((lego_samples, object_samples))
    lego_labels.extend(object_labels)
    all_labels=lego_labels
    
    lego_obj_set = ClassifierDataset(all_imgs=all_samples, all_labels=all_labels)
    train_set, test_set = data.random_split(lego_obj_set, [int(round(len(all_labels)*0.8)), int(round(len(all_labels)*0.2))])
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_scene_data(batch_size):
    lego_samples = numpy_to_torch("lego_scene_arrays")
    object_samples = numpy_to_torch("flickr_scene_arrays")

    num_samples = min(len(lego_samples), len(object_samples))
    num_samples_round = num_samples // batch_size
    lego_samples = lego_samples[:num_samples_round*batch_size]
    object_samples = object_samples[:num_samples_round*batch_size]
    lego_real_set = ImageDataset(lego_objects=lego_samples, real_objects=object_samples, lego_label=[0,1], real_label=[1,0])

    training_set, testing_set = data.random_split(lego_real_set, [int(round(len(lego_real_set)*0.8)), int(round(len(lego_real_set)*0.2))])
    
    training_loader = DataLoader(dataset=lego_real_set, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False)
    print('converted to dataloaders!')
    return training_loader, testing_loader
