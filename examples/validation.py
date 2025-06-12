"""
This module is used to generate the results that were presented for this project.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys

from datetime import datetime
from tqdm import tqdm

from network.visualization import generate_free_energy_graph
from examples.initialization import init_catch_simulation
from simulation.visualization import generate_gif, generate_success_rate_graph, generate_avg_success_rate_graph

NUMBER_SIMULATIONS = 8
NUMBER_ITERATIONS = 10000

DECAY_COEFFICIENT = 0.01875
EXPLORATION_RATE = 0.0003
STRENGTHENING_RATE = 1.009
CONTROLLER_THRESHOLD = 0.40

REGION_SIZES = {'afferent': 16, 'efferent': 48, 'internal': 1024, 'sensory': 1}

TARGET_SUCCESS_RATE = 0.7

if __name__ == '__main__':
    arguments = sys.argv
    ball_speed_orientation = arguments[1]

    validation_dir = f"validation_{ball_speed_orientation}"

    if arguments[2] in ['--avg', '-a']:
        simulations = []
        for i in range(NUMBER_SIMULATIONS):
            simulation_name = os.path.join(validation_dir, f"Catch{i + 1:02d}_{DECAY_COEFFICIENT:.04f}_{EXPLORATION_RATE:.05f}_{STRENGTHENING_RATE:.04f}_{CONTROLLER_THRESHOLD:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}")

            simulation = init_catch_simulation(ball_speed_orientation, DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, simulation_name, REGION_SIZES)
            simulation_dir = simulation.get_simulation_dir()

            for _ in tqdm(range(NUMBER_ITERATIONS), desc=f"processing simulation {i + 1}"):
                simulation.step()

            print("\tgenerating gif...")
            generate_gif(simulation, frame_duration=10)
            print("\tgenerating free_energy_graph...")
            generate_free_energy_graph(simulation.network, simulation_dir)
            print("\tgenerating success_rate_graph...")
            generate_success_rate_graph(simulation)
            print("\tsaving data files...")
            simulation.save_success_history()
            print(f"\n\tsimulation's score : {simulation.get_average_success_rate()}\n")

            simulations.append(simulation)

        print("generating average success rate graph")
        generate_avg_success_rate_graph(simulations, os.path.join("results", validation_dir))

    elif arguments[2] in ["--long", "-l"]:
        long_simulation_name = os.path.join(validation_dir, f"Long_Catch_{DECAY_COEFFICIENT:.04f}_{EXPLORATION_RATE:.05f}_{STRENGTHENING_RATE:.04f}_{CONTROLLER_THRESHOLD:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}")
        long_simulation = init_catch_simulation(ball_speed_orientation, DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, long_simulation_name, REGION_SIZES)

        for _ in tqdm(range(10 * NUMBER_ITERATIONS), desc="processing simulation long simulation"):
            long_simulation.step()

        print("\tgenerating success_rate_graph...")
        generate_success_rate_graph(long_simulation, TARGET_SUCCESS_RATE)
        print(f"\tSimulation's success rate : {long_simulation.get_average_success_rate()}")
        print("\tsaving data files...")
        long_simulation.save_success_history()
    
