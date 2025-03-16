import torch
import pygame
import gymnasium as gym

select_device = "cpu"

if select_device in ["mps", "cuda", "cpu"]: device = torch.device(select_device)
else: device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def get_last_frames(surface, width, height):
    #For now, 1 frame
    pygame.image.save(surface, "./frames/last_frame.png")
    image = pygame.image.load("./frames/last_frame.png")

    downscale = pygame.transform.scale(image, (width, height))
    pygame.image.save(downscale, "./frames/last_frame_downscale.png")

def grey_scale(image):
    arr = pygame.surfarray.arr  