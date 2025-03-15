from network.graph_generation import fixed_average_transmission, self_referring_fixed_average_transmission
from network.region import Region
from network.network import Network

from simulation.geometry.point import Point

import numpy as np

if __name__ == '__main__':
    nb_topographic_regions = 8

    sensory_region_names = [f's{i}' for i in range(nb_topographic_regions)]
    afferent_region_names = [f'a{i}' for i in range(nb_topographic_regions)]
    internal_region_names = ['i0',]
    efferent_region_names = ['e0', 'e1',]

    network_external_regions_names = sensory_region_names
    network_internal_regions_names = afferent_region_names + internal_region_names + efferent_region_names
    network_regions_names = network_external_regions_names + network_internal_regions_names

    sensory_region_size = 1
    afferent_region_size = 4
    internal_region_size = 64
    efferent_region_size = 12
    
    regions = [
        Region(name='s0', size=1, internal=False),
        Region(name='s1', size=1, internal=False),
        Region(name='s2', size=1, internal=False),
        Region(name='s3', size=1, internal=False),
        Region(name='s4', size=1, internal=False),
        Region(name='s5', size=1, internal=False),
        Region(name='s6', size=1, internal=False),
        Region(name='s7', size=1, internal=False),
        Region(name='a0', size=4, internal=True),
        Region(name='a1', size=4, internal=True),
        Region(name='a2', size=4, internal=True),
        Region(name='a3', size=4, internal=True),
        Region(name='a4', size=4, internal=True),
        Region(name='a5', size=4, internal=True),
        Region(name='a6', size=4, internal=True),
        Region(name='a7', size=4, internal=True),
        Region(name='i0', size=64, internal=True),
        Region(name='e0', size=12, internal=True),
        Region(name='e1', size=12, internal=True),
    ]

    connectome_generator = np.random.default_rng()

    sensory_to_afferent_transmission_average = 0.7
    afferent_to_afferent_transmission_average = 0.1
    afferent_to_efferent_transmission_average = 0.05
    afferent_to_internal_transmission_average = 0.5
    afferent_to_self_transmission_average = 0.2
    efferent_to_afferent_transmission_average = 0.05
    efferent_to_efferent_transmission_average = 0.2
    efferent_to_internal_transmission_average = 0.5
    efferent_to_self_transmission_average = 0.1
    internal_to_afferent_transmission_average = 0.4
    internal_to_efferent_transmission_average = 0.4
    internal_to_self_transmission_average = 0.5

    sensory_to_afferent = fixed_average_transmission(sensory_to_afferent_transmission_average, connectome_generator)
    afferent_to_afferent = fixed_average_transmission(afferent_to_afferent_transmission_average, connectome_generator)
    afferent_to_efferent = fixed_average_transmission(afferent_to_efferent_transmission_average, connectome_generator)
    afferent_to_internal = fixed_average_transmission(afferent_to_internal_transmission_average, connectome_generator)
    afferent_to_self = self_referring_fixed_average_transmission(afferent_to_self_transmission_average, connectome_generator)
    efferent_to_afferent = fixed_average_transmission(efferent_to_afferent_transmission_average, connectome_generator)
    efferent_to_efferent = fixed_average_transmission(efferent_to_efferent_transmission_average, connectome_generator)
    efferent_to_internal = fixed_average_transmission(efferent_to_internal_transmission_average, connectome_generator)
    efferent_to_self = self_referring_fixed_average_transmission(efferent_to_self_transmission_average, connectome_generator)
    internal_to_afferent = fixed_average_transmission(internal_to_afferent_transmission_average, connectome_generator)
    internal_to_efferent = fixed_average_transmission(internal_to_efferent_transmission_average, connectome_generator)
    internal_to_self = self_referring_fixed_average_transmission(internal_to_self_transmission_average, connectome_generator)

    afferent_graph_generation_fns = [(afferent_region_names, afferent_to_afferent), (internal_region_names)]
    afferent_regions_connections = {name: {region_name: graph_generation_fn if region_name != name else afferent_to_self for region_names, graph_generation_fn in afferent_graph_generation_fns for region_name in region_names} for name in afferent_region_names}

    regions_connectome = {
        's0': {'a0': sensory_to_afferent},
        's1': {'a1': sensory_to_afferent},
        's2': {'a2': sensory_to_afferent},
        's3': {'a3': sensory_to_afferent},
        's4': {'a4': sensory_to_afferent},
        's5': {'a5': sensory_to_afferent},
        's6': {'a6': sensory_to_afferent},
        's7': {'a7': sensory_to_afferent},
        'a0': 
        'a1':
        'a2':
        'a3':
        'a4':
        'a5':
        'a6':
        'a7':
        'i0':
        'e0':
        'e1':
    }
            