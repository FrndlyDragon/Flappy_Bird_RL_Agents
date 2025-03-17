import pygame
import numpy as np

window_width = 350
window_height = 600

fps = 32

g = 1000

vx = 200  ## velocity at x direction (in the game universe, no need to be shown)

class Vector2f():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def copy(self):
        return Vector2f(x=self.x, y=self.y)

    def as_tuple(self):
        return (self.x, self.y)
    
class Hitbox(pygame.sprite.Sprite):
    def __init__(self, pos, size, hitbox_multiplier=1., col=(255, 0, 0), offsets=(0,0)):
        self.color = col
        self.top = pos[1] + size[1]*(1-hitbox_multiplier)/2 + offsets[1]
        self.bottom = pos[1] + size[1]*(1+hitbox_multiplier)/2 + offsets[1]
        self.left = pos[0] - size[0]*(1-hitbox_multiplier)/2 + offsets[0]
        self.right = pos[0] + size[0]*(1+hitbox_multiplier)/2 + offsets[0]
    
    def __repr__(self):
        return f"Hitbox Object <top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right}>"
    
    def update(self, dx, dy):
        self.top += dy
        self.bottom += dy
        self.left += dx 
        self.right += dx
    
    def blit(self, screen):
        pygame.draw.rect(screen, self.color, (self.left, self.top, self.right-self.left,self.bottom-self.top), width=1)

    def compute_distance_from_point(self, x, y):
        min_dist = np.inf
        between_x = self.left < x and x < self.right
        between_y = self.top < y and y < self.top
        if between_x and between_y: min_dist = min(x - self.left, self.right - x, y - self.top, self.bottom - y)
        elif between_x: min_dist = min(y - self.top, self.bottom - y)
        elif between_y: min_dist = min(x - self.left, self.right - x)
        else:
            xtop = (x - self.top)**2
            xbottom = (x - self.bottom)**2
            yleft = (y - self.left)**2
            yright = (y - self.right)**2
            min_dist = np.sqrt(min(xtop + yleft, xtop + yright, xbottom + yleft, xbottom + yright))
        return min_dist

    def collide(self, hitbox):
        return not (self.top > hitbox.bottom or self.bottom < hitbox.top or self.left > hitbox.right or self.right < hitbox.left)
    
def RenderText(screen, text, col=(255,255,255), pos=(0,0)):
    font = pygame.font.SysFont("Arial" , 18 , bold=True)
    text = font.render(text, 1, pygame.Color(col))
    screen.blit(text, pos)
