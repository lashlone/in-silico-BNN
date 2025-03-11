"""
Visualization module. This module contains functions to visualize the simulation's elements and their evolution.
"""

import os
from PIL import Image, ImageDraw

from simulation.geometry.circle import Circle
from simulation.geometry.point import Point
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.shape import Shape
from simulation.geometry.triangle import IsoscelesTriangle

def create_frame(svg_width: int, svg_height: int, shapes: list[Shape], positions: list[Point]) -> Image.Image:
    """This function creates a Pillow Image object from a list of shapes and their positions."""
    img = Image.new("RGB", (svg_width, svg_height), "#000000")
    draw = ImageDraw.Draw(img)
    
    for shape, pos in zip(shapes, positions):            
        if isinstance(shape, Circle):
            draw.ellipse((pos.x - shape.radius, pos.y - shape.radius, pos.x + shape.radius, pos.y + shape.radius), outline=shape.stroke, width=2, fill=shape.fill)

        elif isinstance(shape, (Rectangle, IsoscelesTriangle)):
            center_movement = pos - shape.center
            polygon_points = [(point.x + center_movement.x, point.y + center_movement.y) for point in shape.get_perimeter_corners()]
            draw.polygon(polygon_points, outline=shape.stroke, width=2, fill=shape.fill)

    return img

def generate_gif(env_history: list[tuple], frame_duration: int, gif_name: str = "env_visualization") -> None:
    """This function creates all the frames represented in the environnement history attribute of a simulation object."""
    # Extracts environnement information, shape information and frames.
    simulation_dir, env_width, env_height = env_history[0]
    shapes = env_history[1] 
    frames = env_history[2:]

    gif_path = os.path.join(simulation_dir, f"{gif_name}.gif")

    frames_img = [create_frame(env_width, env_height, shapes, frame) for frame in frames]
    frames_img[0].save(gif_path, save_all=True, append_images=frames_img[1:], frame_duration=frame_duration, loop=0)

