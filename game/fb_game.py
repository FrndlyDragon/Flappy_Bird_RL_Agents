import random
import sys
import pygame

window_height = 600
window_width = 450
fps = 60

def main():
    pygame.init()
    pygame.display.set_caption('Flappy Bird Game') 

    window = pygame.display.set_mode((window_height, window_width))