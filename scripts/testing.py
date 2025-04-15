"""
This module is used for testing purposes during the building of the project.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys

from datetime import datetime
from tqdm import tqdm

from network.visualization import get_standard_layout, draw_network, generate_free_energy_graph
from scripts.initialization import init_catch_simulation
from simulation.geometry.point import Point
from simulation.visualization import generate_gif, generate_success_rate_graph

CONTROLLER_THRESHOLD = 0.4

def batch_testing():
    for i in range(5):
        coefficient_testing(0.01875, 0.0003, 1.009, CONTROLLER_THRESHOLD)
    # for decay_coefficient in [0.01875, 0.019, 0.02]:
    #     for strengthening_rate in [1.009, 1.01, 1.0125]:
    #         for exploration_rate in [0.0003, 0.00033, 0.000375]:
    #             coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, CONTROLLER_THRESHOLD)

def coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, controller_threshold):
    
    simulation_name = f"Catch_{decay_coefficient:.04f}_{exploration_rate:.05f}_{strengthening_rate:.04f}_{controller_threshold:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

    ball_initial_position = Point(300.0, 160.0)
    ball_x_speed = 2.0
    ball_seed_orientation = 156.2
    simulation = init_catch_simulation(ball_initial_position, ball_x_speed, ball_seed_orientation, decay_coefficient, exploration_rate, strengthening_rate, controller_threshold, simulation_name)
    simulation_dir = simulation.get_simulation_dir()
    
    for _ in tqdm(range(200), desc="processing simulation"):
        simulation.step()

    print("generating gif...")
    generate_gif(simulation, frame_duration=25)
    # print("generating free_energy_graph...")
    # generate_free_energy_graph(simulation.network, simulation_dir)
    # print("generating success_rate_graph...")
    # generate_success_rate_graph(simulation)
    # print("saving data files...")
    # simulation.save_env_history()
    # simulation.network.save_free_energy_history(simulation_dir)
    # simulation.save_success_history()

    if simulation.network.get_size() == 128:
        layout = get_standard_layout()
        for i in tqdm(range(5), desc = "generating network frames"):
            figure = draw_network(simulation.network, layout)
            figure.savefig(os.path.join(simulation_dir, f"frame{i:03d}.png"))
            simulation.step()

if __name__ == "__main__":
    arguments = sys.argv

    if arguments[1] == "-batch":
        batch_testing()

    if arguments[1] == "-coef":
        decay_coefficient = float(arguments[2])
        exploration_rate = float(arguments[3])
        strengthening_rate = float(arguments[4])
        coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, CONTROLLER_THRESHOLD)
    