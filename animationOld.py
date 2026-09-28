import pygame
import numpy as np
import time
import matplotlib
matplotlib.use("QtAgg") # or "Qt5Agg" if you have PyQt5 installed  
import matplotlib.pyplot as plt
from simulation import Simulation

# Initialize Pygame and Font module
pygame.init()
pygame.font.init()



width = 1200
height = 800
scaling = 800
screen = pygame.display.set_mode((width, height))

# Setup a font for drawing the FPS text
font = pygame.font.SysFont("Arial", 24)

# Create an instance of your Simulation class
sim = Simulation(n=500)

clock = pygame.time.Clock()
running = True

plt.ion()  # interactive mode — non-blocking
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel("Frame")
ax.set_ylabel("Total Kinetic Energy")

energy_history = []
frame_count = 0

# Main simulation loop
for _ in range(sim.frames):
  # Handle window events
  for event in pygame.event.get():
    if event.type == pygame.QUIT or (
        event.type == pygame.KEYDOWN
        and (event.key == pygame.K_q or event.key == pygame.K_ESCAPE)
    ):
      running = False

  if not running:
    break

  # 1. Compute the next frame
  positions = sim.next_frame()

  # 2. Scale positions to match the window dimensions
  scaled_positions = positions * np.array([width, height, 1])

  # 3. Clear the screen
  screen.fill((0, 0, 0))

  # 4. Draw the particles
  for x, y, z in scaled_positions:
    pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), np.mean(np.array([width, height]))*sim.collision_distance)

  # 5. Render the FPS text
  # clock.get_fps() returns a float, we convert it to an integer for display
  fps_text = font.render(
      f"FPS: {int(clock.get_fps())}", True, (255, 255, 0)
  )
  screen.blit(fps_text, (10, 10))  # Draw in the top-left corner

  # 6. Update the display
  pygame.display.flip()

  total_ke = 0.5 * np.sum(sim.mass_arr[:, None] * sim.vel_arr**2)
  energy_history.append(total_ke)
  frame_count += 1

    # update the plot every N frames (updating every frame is slow)
  if frame_count % 10 == 0:
      line.set_data(range(len(energy_history)), energy_history)
      ax.relim()
      ax.autoscale_view()
      fig.canvas.draw()
      fig.canvas.flush_events()

  # 7. Tick the clock to control the frame rate and track time
  clock.tick(
      300
  )  # Caps the game at 60 FPS (remove or adjust if you want it uncapped)

pygame.quit()
plt.ioff()
plt.show()

