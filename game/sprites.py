import pygame
import numpy as np

from game.utils import Vector2f, window_width, window_height, g, pipe_y_sep, vx, RenderText, Hitbox

class BaseSprite(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.pos = Vector2f(0, 0)
    
    def blit(self, screen):
        screen.blit(self.image, self.pos.as_tuple())

    def update(self, dt, keydown, sprites):
        return False

class Background(BaseSprite):
    def __init__(self):
        self.pos = Vector2f(0, 0)
        self.image = pygame.image.load("assets/background.png")
        self.size = self.image.get_size()

class Ground(BaseSprite):
    def __init__(self):
        self.pos = Vector2f(0, 550)
        self.image = pygame.image.load("assets/ground.png") # patten length is 23
        self.size = self.image.get_size()
    
    def update(self, dt, keydown, sprites):
        self.pos.x = (self.pos.x - vx*dt) % 23 - 23

class Bird(BaseSprite):
    def __init__(self):
        self.alive = True
        self.pos = Vector2f(50, 250)
        self.v = Vector2f(0, 0)

        self.angle = 0

        self._image = pygame.image.load("assets/flappy_bird.png")
        self._size = self._image.get_size()
        self.base_image = pygame.transform.scale(self._image, (int(self._size[0]*0.1), int(self._size[1]*0.1)))
        self.size = self.base_image.get_size()
        self.image = pygame.transform.rotate(self.base_image, -self.angle)

        self.hitbox = Hitbox(self.pos.as_tuple(), self.size, hitbox_multiplier=0.95)
    
    def update(self, dt, keydown, sprites):
        self.pos.y += self.v.y*dt
        self.hitbox.update(0, self.v.y*dt)
        self.v.y += g*dt

        if keydown: self.v.y = -350

        self.angle = 180*np.arctan(self.v.y/(3*vx))/np.pi
        self.image = pygame.transform.rotate(self.base_image, -self.angle)

        self.alive = self.pos.y + self.size[1] < window_height - 30 and self.pos.y > 0
        for pipe in sprites['pipes'].top_pipes:
            #print(self.hitbox, pipe.hitbox, self.hitbox.collide(pipe.hitbox))
            self.alive &= not self.hitbox.collide(pipe.hitbox)
        for pipe in sprites['pipes'].bottom_pipes:
            self.alive &= not self.hitbox.collide(pipe.hitbox)
        return not self.alive

class Pipe(BaseSprite):
    def __init__(self, top=False, yoffset=None): 
        self.width = 89
        self.height = 611
        self.yrange = [225, 525]
        if yoffset is None: self.yoffset = np.random.randint(self.yrange[0], self.yrange[1]+1) + (-pipe_y_sep - self.height) * (not top)
        else: self.yoffset = yoffset
        self.pos = Vector2f(window_width, -self.yoffset)

        self._image = pygame.image.load("assets/pipe.png")
        self._size = self._image.get_size()
        self.base_image = pygame.transform.scale(self._image, (int(self._size[0]*0.3), int(self._size[1]*0.3)))
        self.size = self.base_image.get_size()
        if top: self.image = self.base_image
        else: self.image = pygame.transform.flip(self.base_image, False, True)

        self.alive = True
        self.hitbox = Hitbox(self.pos.as_tuple(), self.size, 0.95)
    
    def update(self, dt, keydown, sprites):
        self.pos.x -= vx*dt
        self.hitbox.update(-vx*dt, 0)
        self.alive = self.pos.x > -self.width 

class Pipes(BaseSprite):
    def __init__(self, no_pipes_time=2, pipe_interval=1.5):
        self.timer = 0
        self.top_pipes = []
        self.bottom_pipes = []

        self.no_pipes_time = no_pipes_time
        self.pipe_interval = pipe_interval
        self.start = False

        self.score = 0

    def update(self, dt, keydown, sprites):
        self.timer += dt
        if self.timer > self.no_pipes_time: self.start = True
        if self.start and self.timer > self.pipe_interval: 
            self.top_pipes.append(Pipe(top=True))
            self.bottom_pipes.append(Pipe(top=False, yoffset=self.top_pipes[-1].yoffset + (-pipe_y_sep - self.top_pipes[-1].height)))
            self.timer = 0
        if self.start and not self.top_pipes[0].alive:
            self.top_pipes.pop(0)
            self.bottom_pipes.pop(0)
            self.score += 1
        for pipe in self.top_pipes:
            pipe.update(dt, keydown, sprites)
        for pipe in self.bottom_pipes:
            pipe.update(dt, keydown, sprites)
    
    def blit(self, screen):
        for pipe in self.top_pipes:
            pipe.blit(screen)
        for pipe in self.bottom_pipes:
            pipe.blit(screen)
        RenderText(screen, f"Score: {self.score}", pos=(0, 20))

class BirdAgenets(BaseSprite):
    def __init__(self, n_agents=16):
        self.bird_sprites = [Bird() for _ in range(n_agents)]