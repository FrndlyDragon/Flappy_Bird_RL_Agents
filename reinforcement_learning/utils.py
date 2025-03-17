import torch
import numpy as np
import pygame
import gymnasium as gym
import cv2
from PIL import Image
from collections import deque

select_device = "cpu"

if select_device in ["mps", "cuda", "cpu"]: device = torch.device(select_device)
else: device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def get_last_frames(surface, width, height, frame_num):
    #For now, 1 frame
    pygame.image.save(surface, f"./frames/last_frame_{frame_num}.png")
    #image = pygame.image.load(f"./frames/last_frame_{frame_num}.png")

    #downscale = pygame.transform.scale(image, (width, height))
    #pygame.image.save(downscale, f"./frames/last_frame_{frame_num}.png")

def process_captures(width, height):
    for i in range(4):
        image = pygame.image.load(f"./frames/last_frame_{i}.png")
        downscale = pygame.transform.scale(image, (width, height))
        arr = pygame.surfarray.array3d(downscale)
        mean_arr = np.mean(arr, axis=2)
        mean_arr3d = mean_arr[..., np.newaxis]
        new_arr = np.repeat(mean_arr3d[:, :, :], 3, axis=2)
        #grayscale_array = np.dot(array[..., :3], [0.2989, 0.587, 0.114])
        new_image = pygame.surfarray.make_surface(new_arr.astype(np.uint8))

        img_width, img_height = new_image.get_size()

        # Define the size of the cropped image
        crop_width = 175
        crop_height = 175

        # Calculate the position to center the crop rectangle
        x = (img_width - crop_width) // 2
        y = (img_height - crop_height) // 2

        cropped = pygame.Rect(x, y, crop_width, crop_height)
        cropped_img = new_image.subsurface(cropped)
        #print(pygame.Surface.get_size(new_image))
        pygame.image.save(cropped_img, f"./frames/last_frame_{i}.png")

def get_frame_input(frame):
    downscale = pygame.transform.scale(frame, (84,84))
    frame = pygame.surfarray.array3d(downscale)
    grayscale_image = frame.mean(axis=2)
    grayscale_image = grayscale_image / 255.0

    print(grayscale_image.shape)
    return grayscale_image

def get_input_buffer():
    buffer = []
    for i in range(4):
        img = pygame.image.load(f"./frames/last_frame_{i}.png")
        frame = get_frame_input(img)
        buffer.append(frame)
    
    #print(buffer)
    frame_stack = np.stack(buffer, axis=0)
    print(frame_stack.shape)
    frame_stack_tensor = torch.tensor(frame_stack).unsqueeze(0).float()
    print(frame_stack_tensor.shape)

    return frame_stack_tensor