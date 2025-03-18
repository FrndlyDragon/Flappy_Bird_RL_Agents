import pygame
import numpy as np
from typing import Literal

from game.utils import fps, RenderText, window_width, window_height, vx
from game.sprites import Background, Bird, Pipe, Pipes, Ground
from game.dynamicRules import DynamicRules

from pygame.locals import FULLSCREEN, DOUBLEBUF
flags = FULLSCREEN | DOUBLEBUF


class FlappyBird():
    def __init__(self, debug_kwargs, agent, state_type:Literal['var','img']='var', max_speed=False):
        self.debug_kwargs = debug_kwargs
        self.state_type = state_type
        self.max_speed = max_speed
        # pygame
        self.screen = pygame.display.set_mode((window_width, window_height)) 
        self.clock = pygame.time.Clock()
        self.timer = 0
        # sprites
        self.background = Background()
        self.ground = Ground()
        self.pipes = Pipes()
        self.bird = Bird()

        self.agent = agent

    def get_sprites(self):
        return [self.background, self.ground, self.pipes, self.bird]
    
    def update(self, dt, action):
        self.ground.update(dt)
        self.pipes.update(dt)
        self.bird.update(dt, action, self.pipes)
        if self.score != self.pipes.passed:
            self.score += 1
    
    def reset(self, flip=True):
        self.previous_score = 0
        self.score = 0
        self.clock = pygame.time.Clock() 
        self.timer = 0
        for sprite in self.get_sprites(): sprite.__init__()
        for sprite in self.get_sprites(): sprite.blit(self.screen, self.debug_kwargs)
        if flip: pygame.display.flip()
        return self.get_state()

    def get_state(self):
         return self.agent.policy.get_input(self)

    def get_reward(self):
        reward = 3/fps + 1*(self.score - self.previous_score) - 1*int(not self.bird.isalive) -0.5*int(self.bird.hit_ground_or_sky)
        self.previous_score = self.score
        return reward

    def step(self, action):
        if self.max_speed: 
            dt = 1/fps
            self.clock.tick()
        else: 
            dt = self.clock.tick(fps)/1000
        self.timer += dt
        pygame.event.get()

        self.update(dt, action)
        for sprite in self.get_sprites(): sprite.blit(self.screen, self.debug_kwargs)

        RenderText(self.screen, f"FPS: {self.clock.get_fps():.1f}")
        RenderText(self.screen, f"Score: {self.score}", pos=(0, 20))
        RenderText(self.screen, f"Timer: {self.timer:.2f}", pos=(0, 40))
        pygame.display.flip()

        kwargs = {'score': self.score}
        return self.get_state(), self.get_reward(), not self.bird.isalive, kwargs
    
    def set_random_state(self, prop_no_pipe=0.25):
        self.reset(False)
        self.ground.update(np.random.randint(0, int(fps*window_width/vx))/fps)

        bird_y_min = 0
        bird_y_max = window_height - self.bird.size[1]

        pipe_width = Pipe().width
        pipe_interval = self.pipes.pipe_interval
        pipe_sep = DynamicRules().random_sep()
        pipe_xs = [np.random.randint(-pipe_width, int(window_width*(1.0 + prop_no_pipe)))]
        while pipe_xs[-1] < window_width: pipe_xs.append(pipe_xs[-1] + int(vx*pipe_interval))
        pipe_xs = pipe_xs[:-1]
        for x in pipe_xs:
            self.pipes.top_pipes.append(Pipe(top=True))
            self.pipes.bottom_pipes.append(Pipe(top=False, yoffset=self.pipes.top_pipes[-1].yoffset + (- pipe_sep - self.pipes.top_pipes[-1].height)))
            self.pipes.top_pipes[-1].pos.x = x
            self.pipes.bottom_pipes[-1].pos.x = x
            if not x > self.bird.hitbox.right and not x + pipe_width < self.bird.hitbox.left: 
                bird_y_min = max(bird_y_min, self.pipes.top_pipes[-1].hitbox.bottom)
                bird_y_max = min(bird_y_max, self.pipes.bottom_pipes[-1].hitbox.top - (self.bird.hitbox.bottom - self.bird.hitbox.top))
        
        self.bird.pos.y = np.random.randint(bird_y_min, bird_y_max)

        bird_y_velocity_max = 350
        self.bird.v.y = np.random.randint(-bird_y_velocity_max, bird_y_velocity_max)
        self.bird.angle = 180*np.arctan(self.bird.v.y/(3*vx))/np.pi
        self.bird.image = pygame.transform.rotate(self.bird.base_image, -self.bird.angle)
        self.bird.image.get_rect().center = self.bird.center

        for sprite in self.get_sprites(): sprite.blit(self.screen, self.debug_kwargs)
        pygame.display.flip()
        return self.get_state()


    """def run(self):
        frames = 0
        running = True
        for sprite_name, sprite in self.sprites.items(): sprite.blit(self.screen, self.debug_kwargs)
        while running: 
            '''if self.max_speed: 
                dt = 1/fps
                self.clock.tick()
            else: 
                dt = self.clock.tick(fps)/1000'''

            events = self.get_event()
            keydown = False
            '''for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False'''
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
            frames += 1

        time.sleep(0.5)"""