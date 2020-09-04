import time 
b = time.time()
import os 
from pathlib import Path
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import argparse
a = time.time()
print('Imports complete in %.2f seconds'%(a-b))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dims = 256
        self.max_res = 512
        self.max_channels = 512
        print('Generator created.')

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.max_res = 512
        self.max_channels = 512
        print(f'Discriminator created.')

class Commander():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device('cpu')
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.iteration_count = 0
        self.latent_dims = self.generator.latent_dims
        self.max_res = self.generator.max_res
        self.max_channels = self.generator.max_channels
        print('Commander object created.')
        print(f'Running on {self.device}.')
    
    def get_batch(self, file_paths, batch_size):
        # makes batches from the dataset
        # TODO : Make a real batch not just random images
        file_paths.sort()
        l = len(file_paths)
        batch = []
        for i in range(l//batch_size):
            batch.append(random.choice(file_paths[(i*batch_size):((i+1)*batch_size)]))
        return batch

    def plot_imgs(self, num_images):
        # plot images on the command line
        # TODO : Make the rand noise fixed for validation
        for _ in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            img = self.decode_img_tensor(self.generator(noise))
            plt.plot(img)
            plt.show()
    
    def make_save_filenames(self, iteration_count):
        # Make generator and discriminator save file_paths
        if iteration_count == 0:
            os.mkdir('./Snapshots')
        os.mkdir(f'./Snapshots/iter_{iteration_count}')
        self.snapshot_path = f'./Snapshots/iter_{iteration_count}'
        gen_file_path = f'./Snapshots/iter_{iteration_count}/gen_{iteration_count}.pt'
        disc_file_path = f'./Snapshots/iter_{iteration_count}/disc_{iteration_count}.pt'
        return (gen_file_path, disc_file_path)
    
    def normalize_img(self, img_array):
        # convert image array [-1,1] to [0,255]
        normalized = (255/2)*(img_array + 1)
        return normalized
    
    def normalize_noise(self, noise):
        # normalizes noise to [-1,1] and returns a tensor
        min_array = np.min(noise)
        max_array = np.max(noise)
        normalized = -1 + (2*(noise-min_array))/(max_array-min_array)
        return torch.tensor(np.asarray(normalized)).float()

    def decode_img_tensor(self, img_tensor):
        # convert image tensor [-1,1] to plottable [0,255]
        img = img_tensor.detach().numpy().reshape(self.max_res, self.max_res, 3)
        img = self.normalize_img(img)
        return img

    def write_images(self, num_images):
        # Writes images to snapshot folder
        # TODO : Make the rand noise fixed for validation
        for i in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            img = self.decode_img_tensor(self.generator(noise))
            cv2.imwrite(f'{self.snapshot_path}/gen_img_{i}.jpg', img)
    
    def log_stats(self, metrics):
        # Logs losses, iteration, learning rate, models_saved status, etc.
        loss_d, loss_g, lr_d, lr_g, model_saved, iter = metrics
        line = '%d => (G: %.6f || D: %.6f) (G: %.8f || D: %.8f) (Models Saved = %s)\n'%(iter, loss_g, loss_d, lr_g, lr_d, model_saved)
        with open('./Snapshots/logs.txt', 'a') as logger:
            logger.write(line)

    def snapshot_model(self):
        # Saves the model and generated images at regular intervals for validation
        generator_save_name, discriminator_save_name = self.make_save_filenames(self.iteration_count)
        torch.save(self.generator, generator_save_name)
        torch.save(self.discriminator, discriminator_save_name)
        self.write_images(20)
        self.log_stats(self.get_metrics()) # TODO : get_metrics func
        print(f'Snapshot taken at {self.iteration_count} iterations.')

    def save_gen_images(self, num_images):
        # To be used after successful training to generate images
        for i in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            # TODO: Normalize noise
            img = self.decode_img_tensor(self.generator(noise))
            cv2.imwrite(f'filename_{i}.jpg', img)
            plt.plot(img)
            plt.show()

    def get_metrics(self):
        # TODO : Get metrics func 
        pass

    def train_gan(self, save_freq=10000):
        # TODO: train_generator
        # TODO: train_discriminator
        if self.iteration_count % save_freq == 0:
            self.snapshot_model()
        self.iteration_count += 1 # Update for every image