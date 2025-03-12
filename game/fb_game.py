import pygame
import time

from game.utils import fps, RenderText, window_width, window_height
from game.sprites import Background, Bird, Pipes, Ground

def run_game(get_event=pygame.event.get, max_time=-1, max_speed=False):
    screen = pygame.display.set_mode((window_width, window_height)) 
    clock = pygame.time.Clock() 
    sprites = {
        'background': Background(),
        'ground': Ground(),
        'pipes': Pipes(),
        'bird': Bird()
    }

    rl_agent = get_event is not pygame.event.get

    running = True
    while running: 
        if max_speed: dt = 1/32 
        else: dt = clock.tick(fps)/1000

        events = get_event()
        keydown = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
        if rl_agent:
            keydown = events  
        else:
            for event in events:
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    keydown = True

        for sprite_name, sprite in sprites.items(): running &= not sprite.update(dt, keydown, sprites)
        for sprite_name, sprite in sprites.items(): sprite.blit(screen)

        RenderText(screen, f"FPS: {clock.get_fps():.1f}")
        pygame.display.flip()

    time.sleep(0.5)