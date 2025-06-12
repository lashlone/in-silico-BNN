"""
This script was used during the poster presentation to demonstrate how the network functions.
"""

import matplotlib
matplotlib.use('Agg')

import os

from datetime import datetime
from tqdm import tqdm

from network.visualization import get_standard_layout, draw_network
from examples.initialization import init_catch_simulation
from simulation.geometry.vector import Vector2D
from simulation.visualization import generate_gif

DECAY_COEFFICIENT = 0.05
EXPLORATION_RATE = 0.01
STRENGTHENING_RATE = 1.1
CONTROLLER_THRESHOLD = 0.40

REGIONS_SIZE = {'afferent': 4, 'efferent': 12, 'internal': 64, 'sensory': 1}

BALL_ACCELERATION = Vector2D(0.0, 0.01)

if __name__ == '__main__':
    while True:
        print("Veuillez choisir un angle pour la simulation, compris entre 150 et 210 degrée.")
        angle = input()
        try:
            if 150.0 <= float(angle) <= 210:
                break
            print(f"\n\tL'angle entré ({angle}) n'est pas compris entre 150 et 210.\n")
        except ValueError as e:
            print(f"\n\tValueError: {e}\n")
    print()

    while True:
        print("Veuillez choisir une direction pour l'accélération verticale (1., 0. ou -1.).")
        acceleration_orientation = input()
        try:
            if float(acceleration_orientation) in [1., 0., -1.]:
                break
            print(f"\n\tLa direction entrée ({acceleration_orientation}) n'est pas valide.\n")
        except ValueError as e:
            print(f"\n\tValueError: {e}\n")
    print()
    
    while True:
        print("Veuillez choisir un nombre de neurones à retirer, compris entre 8 et 24.")
        number_neurons = input()
        try:
            if 8 <= int(number_neurons) <= 24:
                break
            print(f"\n\tLe nombre entré ({number_neurons}) n'est pas compris entre 8 et 24.\n")
        except ValueError as e:
            print(f"\n\tValueError: {e}\n")

    print("\nVeuillez choisir un nom pour la simulation.")
    name = input()
    print()

    simulation_name = f"Demo_{name}_{angle}_{acceleration_orientation}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

    simulation = init_catch_simulation(angle, DECAY_COEFFICIENT, EXPLORATION_RATE, STRENGTHENING_RATE, CONTROLLER_THRESHOLD, simulation_name, REGIONS_SIZE)
    simulation_dir = simulation.get_simulation_dir()

    simulation.ball.set_state(acceleration=acceleration_orientation * BALL_ACCELERATION)

    layout = get_standard_layout()
    simulation.step()
    for i in tqdm(range(4), desc = "generating network frames"):
        simulation.step()
        figure = draw_network(simulation.network, layout)
        figure.savefig(os.path.join(simulation_dir, f"00_init_frame{i:03d}.png"))

    for _ in tqdm(range(750), desc = "progressing simulation"):
        simulation.step()
    
    simulation.reset_agent_position()
    simulation.regenerate_ball()
    simulation.ball_sensory_signal_translator.reset_timer()
    simulation.step()
    for i in tqdm(range(4), desc = "generating network frames"):
        simulation.step()
        figure = draw_network(simulation.network, layout)
        figure.savefig(os.path.join(simulation_dir, f"01_final_frame{i:03d}.png"))
    
    print("removing neurons...")
    simulation.network.remove_neurons(int(number_neurons), "i0", simulation._generator_)

    simulation.reset_agent_position()
    simulation.regenerate_ball()
    simulation.ball_sensory_signal_translator.reset_timer()
    simulation.step()
    for i in tqdm(range(4), desc = "generating network frames"):
        simulation.step()
        figure = draw_network(simulation.network, layout)
        figure.savefig(os.path.join(simulation_dir, f"02_disturbed_frame{i:03d}.png"))
    
    for _ in tqdm(range(750), desc = "progressing simulation"):
        simulation.step()

    simulation.reset_agent_position()
    simulation.regenerate_ball()
    simulation.ball_sensory_signal_translator.reset_timer()
    simulation.step()
    for i in tqdm(range(4), desc = "generating network frames"):
        simulation.step()
        figure = draw_network(simulation.network, layout)
        figure.savefig(os.path.join(simulation_dir, f"03_recovered_frame{i:03d}.png"))

    print("generating environnement's visualization...")
    generate_gif(simulation, 25)
