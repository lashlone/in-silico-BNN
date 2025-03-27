import os
import sys

from datetime import datetime

from demonstration.initialization import init_simulation
from network.visualization import get_standard_layout, draw_network, generate_free_energy_graph
from simulation.visualization import generate_gif

if __name__ == "__main__":
    arguments = sys.argv
    decay_coefficient = float(arguments[1])
    exploration_rate = float(arguments[2])
    strengthening_rate = float(arguments[3])
    simulation_name = f"Pong_{decay_coefficient:.02f}_{exploration_rate:.02f}_{strengthening_rate}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

    simulation = init_simulation(decay_coefficient, exploration_rate, strengthening_rate, simulation_name)
    simulation_dir = simulation.get_simulation_dir()
    for _ in range(1000):
        simulation.step()

    generate_gif(simulation.get_env_history(), frame_duration=25)
    generate_free_energy_graph(simulation.network, simulation_dir)

    layout = get_standard_layout()
    for i in range(5):
        figure = draw_network(simulation.network, layout)
        figure.savefig(os.path.join(simulation_dir, f"frame{i:03d}.png"))
        simulation.step()