import pygame

window_width = 350
window_height = 600

fps = 32

g = 1000

pipe_y_sep = 175

vx = 200  ## velocity at x direction (in the game universe, no need to be shown)

class Vector2f():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def copy(self):
        return Vector2f(x=self.x, y=self.y)

    def as_tuple(self):
        return (self.x, self.y)
    
class Hitbox():
    def __init__(self, pos, size, hitbox_multiplier=1):
        self.top = pos[1] + size[1]*(1-hitbox_multiplier)/2
        self.bottom = pos[1] + size[1]*(1+hitbox_multiplier)/2
        self.left = pos[0] - size[0]*(1-hitbox_multiplier)/2
        self.right = pos[0] + size[0]*(1+hitbox_multiplier)/2
    
    def __repr__(self):
        return f"Hitbox Object <top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right}>"
    
    def update(self, dx, dy):
        self.top += dy
        self.bottom += dy
        self.left += dx 
        self.right += dx

    def collide(self, hitbox):
        return not (self.top > hitbox.bottom or self.bottom < hitbox.top or self.left > hitbox.right or self.right < hitbox.left)
    
def RenderText(screen, text, col=(255,255,255), pos=(0,0)):
    font = pygame.font.SysFont("Arial" , 18 , bold=True)
    text = font.render(text, 1, pygame.Color(col))
    screen.blit(text, pos)
