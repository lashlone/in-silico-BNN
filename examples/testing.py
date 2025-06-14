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
from examples.initialization import init_catch_simulation, init_random_pong_simulation
from simulation.visualization import generate_gif, generate_success_rate_graph


DECAY_COEFFICIENT = 0.01875
EXPLORATION_RATE = 0.0003
STRENGTHENING_RATE = 1.009
CONTROLLER_THRESHOLD = 0.40

REGION_SIZES = {'afferent': 16, 'efferent': 48, 'internal': 1024, 'sensory': 1}

def batch_testing():
    for i in range(5):
        coefficient_testing(0.01875, 0.0003, 1.009, CONTROLLER_THRESHOLD)
    # for decay_coefficient in [0.01875, 0.019, 0.02]:
    #     for strengthening_rate in [1.009, 1.01, 1.0125]:
    #         for exploration_rate in [0.0003, 0.00033, 0.000375]:
    #             coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, CONTROLLER_THRESHOLD)

def coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, controller_threshold):
    
    simulation_name = f"Catch_{decay_coefficient:.04f}_{exploration_rate:.05f}_{strengthening_rate:.04f}_{controller_threshold:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

    ball_seed_orientation = 156.2
    simulation = init_catch_simulation(ball_seed_orientation, decay_coefficient, exploration_rate, strengthening_rate, controller_threshold, simulation_name, REGION_SIZES)
    simulation_dir = simulation.get_simulation_dir()
    
    for _ in tqdm(range(1000), desc="processing simulation"):
        simulation.step()

    print("generating gif...")
    generate_gif(simulation, frame_duration=25)
    print(simulation.get_average_success_rate())
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

    if arguments[1] in ["--batch", "-b"]:
        batch_testing()

    elif arguments[1] in ["--coef", "-c"]:
        decay_coefficient = float(arguments[2])
        exploration_rate = float(arguments[3])
        strengthening_rate = float(arguments[4])
        coefficient_testing(decay_coefficient, exploration_rate, strengthening_rate, CONTROLLER_THRESHOLD)

    elif arguments[1] in ["--random", "-r"]:
        simulation = init_random_pong_simulation(DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, None, REGION_SIZES)

        for _ in tqdm(range(100), desc="processing simulation"):
            simulation.step()

        print("generating gif...")
        generate_gif(simulation, frame_duration=25)
        

    