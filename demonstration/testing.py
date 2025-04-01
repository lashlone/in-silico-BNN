import os
import sys

from datetime import datetime
from tqdm import tqdm

from demonstration.initialization import init_simulation
from network.visualization import get_standard_layout, draw_network, generate_free_energy_graph
from simulation.visualization import generate_gif, generate_success_rate_graph

if __name__ == "__main__":
    arguments = sys.argv
    decay_coefficient = float(arguments[1])
    exploration_rate = float(arguments[2])
    strengthening_rate = float(arguments[3])
    simulation_name = f"Pong_{decay_coefficient:.02f}_{exploration_rate:.03f}_{strengthening_rate}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

    simulation = init_simulation(decay_coefficient, exploration_rate, strengthening_rate, simulation_name)
    simulation_dir = simulation.get_simulation_dir()
    
    for _ in tqdm(range(5000), desc="processing simulation"):
        simulation.step()

    print("generating gif ...")
    generate_gif(simulation, frame_duration=25)
    print("generating free_energy_graph ...")
    generate_free_energy_graph(simulation.network, simulation_dir)
    print("generating success_rate_graph ...")
    generate_success_rate_graph(simulation)
    
    if simulation.network.get_size() == 128:
        layout = get_standard_layout()
        for i in tqdm(range(5), desc = "generating network frames"):
            figure = draw_network(simulation.network, layout)
            figure.savefig(os.path.join(simulation_dir, f"frame{i:03d}.png"))
            simulation.step()