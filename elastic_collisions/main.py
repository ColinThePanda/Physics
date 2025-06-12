import sys
import os
import json

# make it be able to import linear algebra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from panda_math.vector import vec2
import pygame
import math
import numpy as np
import time


class Circle:
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        color: pygame.Color | tuple,
        mass: float = None,
    ):
        self.pos = vec2(x, y)
        self._radius = radius
        self.velocity: vec2 = vec2(0, 0)
        if mass is None:
            self.mass = math.pi * radius * radius * 1  # Area * density
        else:
            self.mass = mass

        if isinstance(color, tuple):
            if not all([value <= 255 and value >= 0 for value in color]):
                print(f"Invalid color for circle {self}")
                return
            self.color = pygame.Color(*color)
        elif isinstance(color, pygame.Color):
            self.color = color
        else:
            print(
                f"Color attribute for circle {self} must be a pygame color object or a tuple"
            )

    def __str__(self):
        return f"Circle(mass={int(self.mass)}, pos=({int(self.pos.x)}, {int(self.pos.y)}), vel=({int(self.velocity.x)}, {int(self.velocity.y)})"

    def __repr__(self):
        return str(self)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, new_value):
        if isinstance(new_value, float) or isinstance(new_value, int):
            self._radius = new_value

    def update_pos(self):
        self.pos += self.velocity * 0.99

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, tuple(self.pos), self.radius)

    def update(self, screen: pygame.Surface):
        self.update_pos()
        self.draw(screen)


class AcurateCircle(Circle):
    def __init__(self, x, y, color, radius=None, mass=None, density=1):
        if radius is None and mass is None:
            raise ValueError("You must provide either a radius or a mass.")

        if radius is not None and mass is None:
            mass = math.pi * radius**2 * density
        elif mass is not None and radius is None:
            radius = math.sqrt(mass / (math.pi * density))

        super().__init__(x, y, radius, color, mass)


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


def center():
    return vec2(screen.get_size()) // 2


def circle_touching(circle1: Circle, circle2: Circle):
    distance = (circle1.pos - circle2.pos).magnitude
    sum_of_radii = circle1.radius + circle2.radius
    return sum_of_radii > distance


def circle_screen_collide(
    circle: Circle, screen_width: int, screen_height: int, restitution=1
):
    """Handle collisions between a circle and the screen boundaries."""
    collision_occurred = False

    # Left boundary collision
    if circle.pos.x - circle.radius < 0:
        circle.velocity.x = abs(circle.velocity.x) * restitution
        circle.pos.x = circle.radius
        collision_occurred = True

    # Right boundary collision
    if circle.pos.x + circle.radius > screen_width:
        circle.velocity.x = -abs(circle.velocity.x) * restitution
        circle.pos.x = screen_width - circle.radius
        collision_occurred = True

    # Top boundary collision
    if circle.pos.y - circle.radius < 0:
        circle.velocity.y = abs(circle.velocity.y) * restitution
        circle.pos.y = circle.radius
        collision_occurred = True

    # Bottom boundary collision
    if circle.pos.y + circle.radius > screen_height:
        circle.velocity.y = -abs(circle.velocity.y) * restitution
        circle.pos.y = screen_height - circle.radius
        collision_occurred = True

    return collision_occurred


def elastic_collide(obj1: Circle, obj2: Circle):
    collision_normal = (obj2.pos - obj1.pos).normalize()
    tangent = vec2(-collision_normal.y, collision_normal.x)

    v1n = obj1.velocity.dot(collision_normal)
    v1t = obj1.velocity.dot(tangent)
    v2n = obj2.velocity.dot(collision_normal)
    v2t = obj2.velocity.dot(tangent)

    m1 = obj1.mass
    m2 = obj2.mass

    # New normal velocities after elastic collision
    v1n_after = ((m1 - m2) * v1n + 2 * m2 * v2n) / (m1 + m2)
    v2n_after = ((m2 - m1) * v2n + 2 * m1 * v1n) / (m1 + m2)

    # Compute velocity scaling based on collision angle
    rel_velocity = obj1.velocity - obj2.velocity
    angle_factor = abs(rel_velocity.normalize().dot(collision_normal))  # [0, 1]

    min_scale = 1.0
    angle_scale = 0.2
    velocity_scale = max(0.8, min_scale - angle_scale * angle_factor)

    # Reconstruct new velocities
    v1n_vec = collision_normal * v1n_after * velocity_scale
    v1t_vec = tangent * v1t
    v2n_vec = collision_normal * v2n_after * velocity_scale
    v2t_vec = tangent * v2t

    obj1.velocity = v1n_vec + v1t_vec
    obj2.velocity = v2n_vec + v2t_vec

    # Positional correction to resolve overlap
    delta = obj2.pos - obj1.pos
    dist = delta.magnitude
    min_dist = obj1.radius + obj2.radius

    if dist < min_dist and dist > 0:
        penetration_depth = min_dist - dist
        correction_dir = delta.normalize()

        total_mass = obj1.mass + obj2.mass
        obj1.pos += correction_dir * (-penetration_depth * (obj2.mass / total_mass))
        obj2.pos += correction_dir * (penetration_depth * (obj1.mass / total_mass))


def load_setup(filename: str) -> tuple[list[Circle] | list, bool]:
    """Load circle configurations from a JSON file."""
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    print(filename)
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(os.path.exists(filename))
        print(f"Error: Could not find file {filename}")
        return [], False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}")
        return [], False

    circles = []
    graph = False

    if "Graph" in data:
        graph = data.get("Graph", "False").lower() == "true"

    # Check if the "Circles" key exists in the JSON
    if "Circles" not in data:
        print("Error: JSON file does not contain a 'Circles' key")
        return [], graph

    for circle_data in data["Circles"]:
        try:
            # Get circle position (relative to center)
            x = float(circle_data.get("x", 0))
            y = float(circle_data.get("y", 0))

            # Get radius if provided
            radius = (
                float(circle_data.get("radius", None))
                if circle_data.get("radius") is not None
                else None
            )

            # Get mass if provided
            mass_str = circle_data.get("mass", "None")
            mass = None if mass_str == "None" else float(mass_str)

            # Get color
            color_data = circle_data.get("color", {"r": 255, "g": 255, "b": 255})
            color = (
                color_data.get("r", 0),
                color_data.get("g", 0),
                color_data.get("b", 0),
            )

            # Get density for accurate circles
            density = float(circle_data.get("density", 1))

            # Get initial velocity
            x_vel = float(circle_data.get("xvel", 0))
            y_vel = float(circle_data.get("yvel", 0))

            # Check if it's an accurate circle
            is_accurate = circle_data.get("Acurate", "False").lower() == "true"

            # Create the appropriate circle type
            if is_accurate:
                # Create circle with position relative to center
                circle = AcurateCircle(
                    x=center().x + x,
                    y=center().y + y,
                    radius=radius,
                    color=color,
                    mass=mass,
                    density=density,
                )
            else:
                circle = Circle(
                    x=center().x + x,
                    y=center().y + y,
                    radius=radius if radius is not None else 10,
                    color=color,
                    mass=mass,
                )

            # Set velocity
            circle.velocity = vec2(x_vel, y_vel)

            circles.append(circle)

        except (ValueError, KeyError) as e:
            print(f"Error processing circle data: {e}")
            print(f"Problematic circle data: {circle_data}")
            continue

    return circles, graph

pygame.init()
screen_size = (800, 600)

BLACK = pygame.Color(0, 0, 0)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)

screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
pygame.display.set_caption("Circle Physics Simulation")
clock = pygame.time.Clock()

# Load circles from JSON file
circles, graph = load_setup("setup.json")

print(circles, graph)
print(type(circles), type(graph))

if graph:
    import matplotlib.pyplot as plt

    for circle in circles:
        circle.draw(screen)
        pygame.display.flip()
    
    plt.ion()
    fig, ax = plt.subplots()
    plt.ylabel("Velocity")
    plt.xlabel("Time")
    plt.title("Velocity over Time")

    velocity_data = [[] for _ in circles]  # One list per circle
    total_velocity_data = []  # List to track the sum of all circle velocities
    time_data = []

    # Create a line for each circle with a unique label and color
    lines = []
    for i, circle in enumerate(circles):
        (line,) = ax.plot([], [], label=f"Circle {i}")
        lines.append(line)

    # Create a line for the total velocity sum
    (total_velocity_line,) = ax.plot(
        [], [], label="Total Velocity", color="black", linestyle="--"
    )

    plt.legend()
    plt.draw()
    plt.pause(0.001)

    start_time = time.time()

time.sleep(1)

printing = True
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Check for collisions between all pairs of circles
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            if circle_touching(circles[i], circles[j]):
                elastic_collide(circles[i], circles[j])

    # Check for screen collisions for all circles
    for circle in circles:
        circle_screen_collide(circle, *screen.get_size())
        circle.update(screen)

    if graph:
        elapsed_time = time.time() - start_time
        time_data.append(elapsed_time)

        # Calculate the sum of magnitudes of velocities for all circles
        total_velocity = sum(circle.velocity.magnitude for circle in circles)
        total_velocity_data.append(total_velocity)

        # Update each circle's velocity data
        for i, circle in enumerate(circles):
            velocity_data[i].append(circle.velocity.magnitude)
            lines[i].set_xdata(time_data)
            lines[i].set_ydata(velocity_data[i])

        # Update the total velocity sum line
        total_velocity_line.set_xdata(time_data)
        total_velocity_line.set_ydata(total_velocity_data)

        # Rescale and redraw
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    if printing:
        os.system("cls" if sys.platform == "win32" else "clear")

        # Debug info
        print(f"Number of circles: {len(circles)}")
        for i, circle in enumerate(circles):
            print(
                f"Circle {i}: mass={circle.mass:.2f}, pos=({circle.pos.x:.1f}, {circle.pos.y:.1f}), vel=({circle.velocity.x:.1f}, {circle.velocity.y:.1f})"
            )
        for i, circle in enumerate(circles, 1):
            for j, circle2 in enumerate(circles, 1):
                if circle == circle2:
                    continue
                print(
                    f"Mass ratio between circle{i} and circle{j} is {circle.mass / circle2.mass}"
                )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
