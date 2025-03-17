import pygame
import numpy as np

from game.utils import Vector2f, window_width, window_height, g, vx, RenderText, Hitbox
from game.dynamicRules import DynamicRules

class BaseSprite(pygame.sprite.Sprite):
    image = None
    hitbox = None
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.pos = Vector2f(0, 0)
    
    def blit(self, screen, debug_kwargs):
        screen.blit(self.image, self.pos.as_tuple())
        if self.hitbox is not None: 
            if debug_kwargs['hitbox_show']: self.hitbox.blit(screen)


class Background(BaseSprite):
    def __init__(self):
        super().__init__()
        self.pos = Vector2f(0, 0)
        self.image = pygame.image.load("assets/background.png").convert_alpha()
        self.size = self.image.get_size()


class Ground(BaseSprite):
    def __init__(self):
        super().__init__()
        self.pos = Vector2f(0, 550)
        self.image = pygame.image.load("assets/ground.png").convert_alpha() # patten length is 23
        self.size = self.image.get_size()
    
    def update(self, dt):
        self.pos.x = (self.pos.x - vx*dt) % 23 - 23


class Bird(BaseSprite):
    def __init__(self, alpha=1):
        super().__init__()
        self.isalive = True
        self.pos_yrand = 10
        self.pos = Vector2f(50, 300 + np.random.randint(-self.pos_yrand, self.pos_yrand+1))
        self.v = Vector2f(0, 0)
        self.angle = 0

        self._image = pygame.image.load("assets/flappy_bird.png").convert_alpha()
        self._image.set_alpha(255*alpha)
        self._size = self._image.get_size()
        self.base_image = pygame.transform.scale(self._image, (int(self._size[0]*0.1), int(self._size[1]*0.1)))
        self.size = self.base_image.get_size()
        self.image = self.base_image
        self.center = self.image.get_rect().center

        self.hitbox = Hitbox(self.pos.as_tuple(), (self.size[0]-10, self.size[1]), hitbox_multiplier=0.95, col=(0, 0, 255, alpha), offsets=(10,5))

        self.score = 0
    
    def update(self, dt, keydown, pipes):
        if not self.isalive: 
            self.pos.x -= vx*dt
            self.hitbox.update(-vx*dt, 0)
            return
        
        self.pos.y += self.v.y*dt
        self.hitbox.update(0, self.v.y*dt)
        self.v.y += g*dt

        if keydown: self.v.y = -350

        self.angle = 180*np.arctan(self.v.y/(3*vx))/np.pi
        self.image = pygame.transform.rotate(self.base_image, -self.angle)
        self.image.get_rect().center = self.center

        self.isalive = self.pos.y + self.size[1] < window_height and self.pos.y > 0
        self.hit_ground_or_sky = not self.isalive
        for pipe in pipes.top_pipes:
            self.isalive &= not self.hitbox.collide(pipe.hitbox)
        for pipe in pipes.bottom_pipes:
            self.isalive &= not self.hitbox.collide(pipe.hitbox)
        self.hit_pipe = not self.hit_ground_or_sky and not self.isalive


class Pipe(BaseSprite):
    def __init__(self, top=False, yoffset=None): 
        super().__init__()
        self.width = 89
        self.height = 611
        self.yrange = [50 + DynamicRules().pipe_y_sep, 350 + DynamicRules().pipe_y_sep]
        if yoffset is None: self.yoffset = np.random.randint(self.yrange[0], self.yrange[1]+1) + (-DynamicRules().pipe_y_sep - self.height) * (not top)
        else: self.yoffset = yoffset
        self.pos = Vector2f(window_width, -self.yoffset)

        self._image = pygame.image.load("assets/pipe.png").convert_alpha()
        self._size = self._image.get_size()
        self.base_image = pygame.transform.scale(self._image, (int(self._size[0]*0.3), int(self._size[1]*0.3)))
        self.size = self.base_image.get_size()
        if top: self.image = self.base_image
        else: self.image = pygame.transform.flip(self.base_image, False, True)

        self.isalive = True
        self.hitbox = Hitbox(self.pos.as_tuple(), self.size, col=(255, 0, 0))

    def update(self, dt):
        self.pos.x -= vx*dt
        self.hitbox.update(-vx*dt, 0)
        self.isalive = self.pos.x > -self.width 


class Pipes(BaseSprite):
    def __init__(self, no_pipes_time=3, pipe_interval=1.5):
        super().__init__()
        self.timer = 0
        self.top_pipes = []
        self.bottom_pipes = []

        self.no_pipes_time = no_pipes_time
        self.pipe_interval = pipe_interval
        self.start = False

        self.passed = 0

    def update(self, dt):
        self.timer += dt
        if self.timer > self.no_pipes_time: 
            self.start = True
        if (self.start and self.timer > self.pipe_interval): 
            self.top_pipes.append(Pipe(top=True))
            self.bottom_pipes.append(Pipe(top=False, yoffset=self.top_pipes[-1].yoffset + (-DynamicRules().pipe_y_sep - self.top_pipes[-1].height)))
            self.timer = 0
        if self.start and not self.top_pipes[0].isalive:
            self.top_pipes.pop(0)
            self.bottom_pipes.pop(0)
            self.passed += 1
        for pipe in self.top_pipes: pipe.update(dt)
        for pipe in self.bottom_pipes: pipe.update(dt)
    
    def blit(self, screen, debug_kwargs):
        for pipe in self.top_pipes: pipe.blit(screen, debug_kwargs)
        for pipe in self.bottom_pipes: pipe.blit(screen, debug_kwargs)

