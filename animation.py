import pygame
import numpy as np
import time
from simulation import animetion_pos_matrix, dt

pygame.init()

screen = pygame.display.set_mode((1200,800))

clock = pygame.time.Clock()
quit = False

for positions in animetion_pos_matrix:
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                quit = True

    if quit:
        break

    screen.fill((0, 0, 0))

    for x, y in positions:
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 3)
    
    time.sleep(dt)


    pygame.display.flip()

pygame.quit()


