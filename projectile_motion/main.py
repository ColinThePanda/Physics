import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from panda_math.vector import vec2
import pygame


def vector_to_angle(vector: vec2):
    angle_rad = math.atan2(vector.y, vector.x)
    angle_deg = math.degrees(angle_rad) % 360
    return angle_deg


def calculate_x(angle_deg: float, velocity: float, time: float):
    angle_rad = math.radians(angle_deg)  # Convert degrees to radians
    return velocity * math.cos(angle_rad) * time


def calculate_y(
    angle_deg: float,
    velocity: float,
    time: float,
    height: float = 0,
    gravity: float = 32.174,
):
    angle_rad = math.radians(angle_deg)
    return height + velocity * math.sin(angle_rad) * time - 0.5 * gravity * time**2


def calculate_pos(
    angle_deg: float,
    velocity: float,
    time: float,
    starting_pos: vec2 = vec2(0, 0),
    gravity: float = 32.174,
):
    x = calculate_x(angle_deg, velocity, time)
    y = calculate_y(angle_deg, velocity, time, 0, gravity)
    return vec2(x, y) + starting_pos


starting_pos = vec2(100, 300)
gravity = 32.174

pygame.init()
screen_size = vec2(800, 600)

BLACK = pygame.Color(0, 0, 0)
BLUE = pygame.Color(0, 0, 255)

screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Projectile Motion")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    
    max_time = 7

    mouse_pos = vec2(pygame.mouse.get_pos())
    mouse_vector = mouse_pos - starting_pos
    direction = mouse_vector.normalize()
    
    dx = mouse_pos.x - starting_pos.x
    dy = (screen_size.y - mouse_pos.y) - starting_pos.y

    t = max_time
    vx = dx / t
    vy = (dy + 0.5 * gravity * t**2) / t

    velocity = math.hypot(vx, vy)
    angle = math.degrees(math.atan2(vy, vx))

    step_size = 0.1
    time = 0.0
    while time <= max_time:
        position = calculate_pos(angle, velocity, time, starting_pos)
        position.y = screen_size.y - position.y
        pygame.draw.circle(screen, BLUE, tuple(position), 5)
        time += step_size

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
