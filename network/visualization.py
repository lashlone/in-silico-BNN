
from network.network import Network
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from matplotlib import patches as mpatches

def get_color(value: float) -> tuple[float, float, float]:
    """Returns a color interpolated between black (#000000) and yellow (#FFFF00)"""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"given value ({value}) must be between 0.0 and 1.0.")
    return (value, value, 0.0)

def get_standard_layout() -> list[tuple[float, float]]:
    """Generates a structured layout for the neurons based on the project's standard network structure."""
    positions = []
    x_offset = 0.0
    y_offset = 3.5
    spacing = 0.3
    group_spacing = 1.0

    # Sensory regions (8 groups of 1 neuron arranged in a vertical column)
    for i in range(8):
        positions.append((x_offset, y_offset - i * group_spacing))

    # Afferent regions (8 groups of 4 neurons arranged in squares)
    x_offset += 0.4
    y_offset = 3.65
    for i in range(8):
        base_x = x_offset
        base_y = y_offset - i * group_spacing
        for j in range(2):
            for k in range(2):
                positions.append((base_x + j * spacing, base_y - k * spacing))

    # Internal region (1 group of 64 neurons arranged in a square)
    x_offset += 1.1
    y_offset = 2.8
    spacing = 0.8
    for i in range(8):
        for j in range(8):
            positions.append((x_offset + i * spacing, y_offset - j * spacing))

    # Efferent regions (2 groups of 12 neurons arranged in 3x4 rectangles)
    x_offset += 6.4
    y_offset = 2.0
    spacing = 0.4
    group_spacing = 2.4
    for i in range(2):
        base_x = x_offset 
        base_y = y_offset - i * group_spacing
        for j in range(12):
            positions.append((base_x + (j % 3) * spacing, base_y - (j // 3) * spacing))

    return positions

def draw_network(network: Network, layout: list[tuple[float, float]], weight_attenuation: float = 5.0) -> Figure:
    """
    Draws a neural network's representation based on a structured layout.
        - network: Network object to draw from.
        - layout: list of position for the network's neurons in the generated graph.
        - weight_attenuation (optional): float coefficient used to show only the main connections.
    """
    if not isinstance(network, Network):
        raise TypeError(f"unsupported parameter type(s) for network: '{type(network).__name__}'")
    if not isinstance(layout, list):
        raise TypeError(f"unsupported parameter type(s) for layout: '{type(layout).__name__}'")
    if not weight_attenuation >= 1.0:
        raise ValueError(f"given attenuation factor ({weight_attenuation}) must be greater than one.")
    if not len(layout) == network._size_:
        raise ValueError(f"given layout's size {len(layout)} does not match the network's size {network._size_}.")
    
    G = nx.DiGraph()
    neuron_states = network.get_state()
    weights = network.get_conformation()
    alpha_values = np.exp(-weight_attenuation * weights)
    num_neurons = len(neuron_states)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')

    edges = []
    edge_alphas = []
    
    # Draws edges first
    for i in range(num_neurons):
        for j in range(num_neurons):
            weight = weights[i, j]
            if not np.isnan(weight): # Only draws existing edges
                G.add_edge(j, i, weight=weight)
                edges.append((j, i))
                edge_alphas.append(alpha_values[i, j])

    nx.draw_networkx_edges(
        G, layout, edgelist=edges, edge_color="white", alpha=edge_alphas, 
        arrowstyle='-|>', arrowsize=8, ax=ax
    )

    # Draws neurons on top
    for i, state in enumerate(neuron_states):
        x, y = layout[i]
        color = get_color(state)
        circle = mpatches.Circle((x, y), radius=0.1, color=color, ec='white', lw=1, zorder=3)
        ax.add_patch(circle)
        G.add_node(i, pos=(x, y))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig

def generate_free_energy_graph(network: Network, simulation_dir: str, file_name: str = "free_energy_evolution"):
    if not isinstance(network, Network):
        raise TypeError(f"unsupported parameter type(s) for network: '{type(network).__name__}'")
    if not os.path.isdir(simulation_dir):
        raise ValueError(f"given simulation directory ({simulation_dir}) does not exist.")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(network._free_energy_history_)
    ax.set_title("Évolution de l'énergie libre du réseau au cours de la simulation.")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Énergie libre")
    ax.set_ylim(-network._size_, network._size_)

    fig.savefig(os.path.join(simulation_dir, f"{file_name}.png"))