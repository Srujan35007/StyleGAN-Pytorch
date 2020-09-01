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
        file_paths.sort()
        l = len(file_paths)
        batch = []
        for i in range(l//batch_size):
            batch.append(random.choice(file_paths[(i*batch_size):((i+1)*batch_size)]))
        return batch

    def plot_imgs(self, num_images):
        for _ in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            img = self.decode_img_tensor(self.generator(noise))
            plt.plot(img)
            plt.show()
    
    def make_save_filenames(self, iteration_count):
        if iteration_count == 0:
            os.mkdir('./Snapshots')
        os.mkdir(f'./Snapshots/iter_{iteration_count}')
        self.snapshot_path = f'./Snapshots/iter_{iteration_count}'
        gen_file_path = f'./Snapshots/iter_{iteration_count}/gen_{iteration_count}.pt'
        disc_file_path = f'./Snapshots/iter_{iteration_count}/disc_{iteration_count}.pt'
        return (gen_file_path, disc_file_path)
    
    def decode_img_tensor(self, img_tensor):
        img = img_tensor.detach().numpy().reshape(self.max_res, self.max_res, 3)
        # TODO: Normalize image
        return img

    def write_images(self, num_images):
        for i in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            img = self.decode_img_tensor(self.generator(noise))
            cv2.imwrite(f'{self.snapshot_path}/gen_img_{i}.jpg', img)

    def snapshot_model(self):
        generator_save_name, discriminator_save_name = self.make_save_filenames(self.iteration_count)
        torch.save(self.generator, generator_save_name)
        torch.save(self.discriminator, discriminator_save_name)
        self.write_images(20)
        # TODO: Log status and metrics
        print(f'Snapshot taken at {self.iteration_count} iterations.')

    def save_gen_images(self, num_images):
        # To be used after successful training
        for i in range(num_images):
            noise = torch.tensor(np.random.normal(0, 0.38, 
                    (1,self.latent_dims))).view(-1, self.latent_dims).float()
            # TODO: Normalize noise
            img = self.decode_img_tensor(self.generator(noise))
            cv2.imwrite(f'filename_{i}.jpg', img)
            plt.plot(img)
            plt.show()

    def train_gan(self, save_freq=10000):
        # TODO: train_generator
        # TODO: train_discriminator
        if self.iteration_count % save_freq == 0:
            self.snapshot_model()
        self.iteration_count += 1 # Update for every image