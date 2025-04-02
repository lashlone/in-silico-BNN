"""
Simulation visualization module. This module contains functions to visualize the simulation's elements and their evolution.
"""

from simulation.geometry.circle import Circle
from simulation.geometry.point import Point
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.shape import Shape
from simulation.geometry.triangle import IsoscelesTriangle
from simulation.base_simulation import Simulation
from simulation.pong import Pong

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw

def create_frame(svg_width: int, svg_height: int, shapes: list[Shape], positions: list[Point]) -> Image.Image:
    """Creates a Pillow Image object from a list of shapes and their positions."""
    img = Image.new("RGB", (svg_width, svg_height), "#000000")
    draw = ImageDraw.Draw(img)
    
    for shape, pos in zip(shapes, positions):            
        if isinstance(shape, Circle):
            draw.ellipse((pos.x - shape.radius, pos.y - shape.radius, pos.x + shape.radius, pos.y + shape.radius), outline=shape.outline, width=2, fill=shape.fill)

        elif isinstance(shape, (Rectangle, IsoscelesTriangle)):
            center_movement = pos - shape.center
            polygon_points = [(point.x + center_movement.x, point.y + center_movement.y) for point in shape.get_perimeter_points()]
            draw.polygon(polygon_points, outline=shape.outline, width=2, fill=shape.fill)

    return img

def generate_gif(simulation: Simulation, frame_duration: int, gif_name: str = "env_visualization") -> None:
    """Generates the frames represented in the environnement history attribute of a simulation object and save them in a animated gif."""
    if not isinstance(simulation, Simulation):
        raise TypeError(f"unsupported parameter type(s) for simulation: '{type(simulation).__name__}'")
    
    env_history = simulation.get_env_history()
    env_width = simulation.width
    env_height = simulation.height

    shapes = env_history[0] 
    frames = env_history[1:]

    gif_path = os.path.join(simulation.get_simulation_dir(), f"{gif_name}.gif")

    frames_img = [create_frame(env_width, env_height, shapes, frame) for frame in frames]
    frames_img[0].save(gif_path, save_all=True, append_images=frames_img[1:], frame_duration=frame_duration, loop=0)

def generate_success_rate_graph(simulation: Pong, file_name: str = "success_rate_evolution", mean_filter_width: int = 8, interpolation_fragment_size: int = 250):
    """Generates the graph of the simulation's success rate evolution."""
    if not isinstance(simulation, Pong):
        raise TypeError(f"unsupported parameter type(s) for simulation: '{type(simulation).__name__}'")
    
    success_history = simulation.get_success_history()

    success_rates = []
    time_stamps = []
    for i in range(len(success_history) + 1 - mean_filter_width):
        success_rate, time_stamp = np.mean(success_history[i: i + mean_filter_width, :], axis=0)
        success_rates.append(success_rate)
        time_stamps.append(time_stamp)

    last_iteration = simulation.get_time()
    linear_interpolation_x = np.linspace(0, last_iteration, last_iteration // interpolation_fragment_size)
    linear_interpolation_y = np.interp(linear_interpolation_x, time_stamps, success_rates)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(time_stamps, success_rates, color='red', label='Taux calculés', zorder=5)
    ax.plot(linear_interpolation_x, linear_interpolation_y, linestyle='--', color='black', label='Interpolation linéaire')    
    ax.set_title("Interpolation de l'évolution du taux de succès au cours de la simulation.")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Taux de succès")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.savefig(os.path.join(simulation.get_simulation_dir(), f"{file_name}.png"))
    plt.close(fig)
    