from linear_algebra.vector import vec2
from linear_algebra.matrix import rotation_matrix_2d, Matrix, shear_matrix_2d
import pygame
import math
import numpy as np
import os
import sys


def draw_arrow(surface, color, start, end, arrow_length=10, arrow_angle=30, width=2):
    # Draw the main line (shaft)
    pygame.draw.line(surface, color, start, end, width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)

    arrow_angle_rad = math.radians(arrow_angle)

    left = (
        end[0] - arrow_length * math.cos(angle - arrow_angle_rad),
        end[1] - arrow_length * math.sin(angle - arrow_angle_rad),
    )
    right = (
        end[0] - arrow_length * math.cos(angle + arrow_angle_rad),
        end[1] - arrow_length * math.sin(angle + arrow_angle_rad),
    )

    pygame.draw.polygon(surface, color, [end, left, right])
    pygame.draw.circle(surface, color, (int(end[0]), int(end[1])), width // 3)


def vector_to_angle(vector: vec2):
    angle_rad = math.atan2(vector.y, vector.x)
    angle_deg = math.degrees(angle_rad) % 360
    return angle_deg


pygame.init()
screen_size = (800, 600)

BLACK = pygame.Color(0, 0, 0)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)

screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
pygame.display.set_caption("Vector and Matrix Demo")
clock = pygame.time.Clock()


def center():
    return vec2(screen.get_size()) // 2


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Get mouse position and calculate direction vector from center()
    mouse_pos = pygame.mouse.get_pos()
    direction = vec2(mouse_pos[0] - center().x, mouse_pos[1] - center().y)

    # Create a normalized vector with length 200
    normalized = direction.normalize() * 200

    # Draw the original normalized vector (RED)
    end_pos = center() + normalized
    draw_arrow(screen, RED, (center().x, center().y), (end_pos.x, end_pos.y), 15, 30, 2)

    # Create rotation matrix and apply it to the normalized vector
    rot = rotation_matrix_2d(np.pi / 2)  # 90 degrees rotation
    rotated_vector = rot * normalized  # Matrix multiplies Vector

    # Draw the rotated vector (GREEN)
    rotated_end_pos = center() + rotated_vector
    draw_arrow(
        screen,
        GREEN,
        (center().x, center().y),
        (rotated_end_pos.x, rotated_end_pos.y),
        15,
        30,
        2,
    )

    # Demonstrate scaling with matrix
    scale_matrix = Matrix([[0.5, 0], [0, 1.5]])  # Scale x by 0.5, y by 1.5
    scaled_vector = scale_matrix * normalized

    # Draw the scaled vector (BLUE)
    scaled_end_pos = center() + scaled_vector
    draw_arrow(
        screen,
        BLUE,
        (center().x, center().y),
        (scaled_end_pos.x, scaled_end_pos.y),
        15,
        30,
        2,
    )

    shearing_matrix = shear_matrix_2d(0.5, 0.5)
    sheared_vector = shearing_matrix * normalized
    shear_end_pos = center() + sheared_vector
    draw_arrow(
        screen,
        YELLOW,
        (center().x, center().y),
        (shear_end_pos.x, shear_end_pos.y),
        15,
        30,
        2,
    )

    os.system("cls" if sys.platform == "win32" else "clear")

    print("Normal:", f"{round(vector_to_angle(normalized/200))}°")
    print("Rotation:", f"{round(vector_to_angle(rotated_vector/200))}°")
    print("Scaled:", f"{round(vector_to_angle(scaled_vector/200))}°")
    print("Sheared:", f"{round(vector_to_angle(sheared_vector/200))}°")

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
