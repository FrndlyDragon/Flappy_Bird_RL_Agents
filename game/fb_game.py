import pygame
import time

from game.utils import fps, RenderText, window_width, window_height
from game.sprites import Background, Bird, Pipes, Ground

from reinforcement_learning.utils import get_last_frames
from pygame.locals import *
flags = FULLSCREEN | DOUBLEBUF

class FlappyBird():
    def __init__(self, debug_kwargs, max_speed=False, max_frames=-1):
        self.debug_kwargs = debug_kwargs
        self.get_event = pygame.event.get
        self.max_speed = max_speed
        self.max_frames = max_frames
        self.rl = False
        self.frame_capture = 0
        self.screen = pygame.display.set_mode((window_width, window_height)) 
        self.clock = pygame.time.Clock() 
        self.sprites = {
            'background': Background(),
            'ground': Ground(),
            'pipes': Pipes(),
            'bird': Bird()
        }
    
    def set_rl(self, get_event, agents):
        self.get_event = get_event
        self.agents = agents
        self.rl = True
        self.sprites['bird'] = self.agents
    
    def reset(self):
        self.frame_capture = 0
        self.clock = pygame.time.Clock() 
        for sprite_name, sprite in self.sprites.items():
            if sprite_name == 'bird' and self.rl: 
                for bird in sprite.birds:
                    bird.__init__(alpha=sprite.bird_alpha)
            else: 
                sprite.__init__()
        if self.rl: self.sprites['bird'] = self.agents
        for sprite_name, sprite in self.sprites.items(): sprite.blit(self.screen, self.debug_kwargs)
        RenderText(self.screen, f"FPS: {self.clock.get_fps():.1f}")
        pygame.display.flip()
    
    def run(self):
        frames = 0
        running = True
        for sprite_name, sprite in self.sprites.items(): sprite.blit(self.screen, self.debug_kwargs)
        while running: 
            if self.max_speed: 
                dt = 1/fps
                self.clock.tick()
            else: 
                dt = self.clock.tick(fps)/1000

            events = self.get_event()
            keydown = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False
            if self.rl:
                keydown = events 
            else:
                for event in events:
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        keydown = True
            if self.rl: self.agents.get_features(self)
            for sprite_name, sprite in self.sprites.items(): running &= not sprite.update(dt, keydown, self.sprites)
            for sprite_name, sprite in self.sprites.items(): sprite.blit(self.screen, self.debug_kwargs)

            RenderText(self.screen, f"FPS: {self.clock.get_fps():.1f}")
            RenderText(self.screen, f"Score: {self.sprites['pipes'].score}", pos=(0, 20))
            RenderText(self.screen, f"Timer: {self.sprites['ground'].timer:.2f}", pos=(0, 40))
            pygame.display.flip()
            if frames > self.max_frames and self.max_frames > 0: break
            get_last_frames(self.screen, window_width // 2, window_height // 2, self.frame_capture)
            self.frame_capture += 1
            if self.frame_capture >= 4:
                self.frame_capture = 0
            frames += 1

        time.sleep(0.5)