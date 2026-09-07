import pygame
import numpy as np
import time
from simulation import animation_pos_matrix, dt


pygame.init()
width = 1200
height = 800
scaled_pos = animation_pos_matrix * np.array([width, height, 1])

#print(animation_pos_matrix)

screen = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
quit = False

for positions in scaled_pos:
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                quit = True

    if quit:
        break

    screen.fill((0, 0, 0))

    for x, y, z in positions:
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 3)
    
    time.sleep(dt)


    pygame.display.flip()

pygame.quit()


