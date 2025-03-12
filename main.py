import pygame
import numpy as np

from game.fb_game import run_game

if __name__ == "__main__":           
    pygame.init() 
    pygame.display.set_caption("Flappy Bird")  
    run_game()