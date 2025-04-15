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
from scripts.initialization import init_catch_simulation
from simulation.geometry.point import Point
from simulation.visualization import generate_gif, generate_success_rate_graph, generate_avg_success_rate_graph

NUMBER_SIMULATIONS = 8
NUMBER_ITERATIONS = 10000

DECAY_COEFFICIENT = 0.01875
EXPLORATION_RATE = 0.0003
STRENGTHENING_RATE = 1.009
CONTROLLER_THRESHOLD = 0.40

BALL_INITIAL_POSITION = Point(300.0, 160.0)
BALL_X_SPEED = 2.0

if __name__ == '__main__':
    arguments = sys.argv
    ball_speed_orientation = arguments[1]

    validation_dir = f"validation_{ball_speed_orientation}"

    if arguments[2] == '--avg':
        simulations = []
        for i in range(NUMBER_SIMULATIONS):
            simulation_name = os.path.join(validation_dir, f"Catch{i + 1:02d}_{DECAY_COEFFICIENT:.04f}_{EXPLORATION_RATE:.05f}_{STRENGTHENING_RATE:.04f}_{CONTROLLER_THRESHOLD:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}")

            simulation = init_catch_simulation(BALL_INITIAL_POSITION, BALL_X_SPEED, ball_speed_orientation, DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, simulation_name)
            simulation_dir = simulation.get_simulation_dir()

            for _ in tqdm(range(NUMBER_ITERATIONS), desc=f"processing simulation {i + 1}"):
                simulation.step()

            print("\tgenerating gif...")
            generate_gif(simulation, frame_duration=10)
            print("\tgenerating free_energy_graph...")
            generate_free_energy_graph(simulation.network, simulation_dir)
            print("\tgenerating success_rate_graph...")
            generate_success_rate_graph(simulation)
            print("saving data files...")
            simulation.save_success_history()

            simulations.append(simulation)
        
        print("generating average success rate graph")
        generate_avg_success_rate_graph(simulations, os.path.join("results", validation_dir))

    elif arguments[2] == "--long":
        long_simulation_name = os.path.join(validation_dir, f"Long_Catch_{DECAY_COEFFICIENT:.04f}_{EXPLORATION_RATE:.05f}_{STRENGTHENING_RATE:.04f}_{CONTROLLER_THRESHOLD:.02f}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}")
        long_simulation = init_catch_simulation(BALL_INITIAL_POSITION, BALL_X_SPEED, ball_speed_orientation, DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, long_simulation_name)

        for _ in tqdm(range(10 * NUMBER_ITERATIONS), desc="processing simulation long simulation"):
            long_simulation.step()

        print("\tgenerating success_rate_graph...")
        generate_success_rate_graph(long_simulation)
    
