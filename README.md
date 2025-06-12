# in-silico-BNN
This project aims to emulate the behavior of organoid intelligence in a numerical environnement. It also aims to compute various definitions of the system’s free energy to analyze how they evolve over time and investigate their correlation with the model’s learning dynamics.

## Getting started

### Prerequisite
* Python 3.10
* GIT

### Installation
To create a local copy of this repository, use the ```git clone``` command with this project's URL:

 ```
 git clone https://github.com/lashlone/in-silico-BNN.git
 ```

Next, install the required Python libraries listed in the [requirements](requirements.txt) file:

 ```
 pip install -r requirements.txt
 ```

It is strongly recommended to create a virtual environnement to ensure package version compatibilities and to isolate dependencies from your system Python environnement. You can use any tool you are comfortable with (e.g., ```virtualenv```, ```conda```). Here's how to do it using Python’s built-in ```venv``` module, which doesn’t require any additional setup:

 ```
 python -m venv <environnement_name>
 source <environment_name>/bin/activate  # On Linux/macOS
 <environment_name>\Scripts\activate     # On Windows
 ```

### Create a network

The [network](\network) module contains all the functions and classes to create, use and analyze a neural network based on the author's model. To create a Network object, you must first define its regions. Regions objects are initialized by their size, name and type. The size of a region corresponds to the number of neurons within them, while their name later allows to identify them in the network. In this preliminary version, region types only influence the math behind the system's free energy computation. However, the author plan to implement more region's type, such as inhibitory or non-spiking neurons, to allow more complexity in the model.

Then, you must define the available connections between the network's regions, represented by the network's connectome. During the simulation, only the connections defined here will be allowed to be generated. Each connection is parameterized by a graph generating function that is used to initialize the network.

The network class also have multiple other named parameters with default values, such as ` recovery_state_energy_ratio ` and ` decay_coefficient `. They can be adjusted by experienced user to get a better control over the network behavior. Their effects over the network are described in details in the project's official report (available on demand, french version only). 

### Run a simulation

The [simulation](\simulation) module contains a simple implementation of the Pong environnement. To initialize a simulation, it is recommended to use the ` init_catch_simulation `, ` init_PID_pong_simulation ` or ` init_random_pong_simulation ` function defined in the [initialization](\example\initialization.py) file. Experienced user can use the class constructor or use the network in simulations of their own.

Then, the simulation is ran by calling the simulation object's `.step()` method over a given number of iterations.

### Analyze your data

The network and the simulation modules both contain a visualization file where functions to generate graphs of the network's or the simulation's evolution are implemented. All of the generated files are then grouped into the [results](\results) repository under the simulation's name.

### Examples

The [validation](\examples\validation.py) file gives a complete example of the implemented pipeline used to analyse the network's performance, from the network initialization to the generation of the graphs. The [demonstration](\examples\demonstration.py) file is another example of a more lucrative use of the network, where it competes in a Pong match against an opponent controlled by a random walker.

## Contributing

If you want to contribute to this project, please contact the author to learn about his intents for the future of this project and the process for submitting your pull requests.

## Versioning

This project use the [SerVer](https://semver.org/) guidelines for versioning. For a list of the available versions and their description, see the list of [tags](docs/tags.md) used in this project.

## Authors

**Vincent Lachapelle** (vincent-2.lachapelle@polymtl.ca)

Project directed by **Réjean Plamondon** and **Tristan Devaux**.

See also the list of [contributors](docs/contributors.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

I would like to thank every experts who kindly accepted to discuss with me over critical points of my model. Their names and contributions are listed below.

* Alain Vinet (neuron model)
* Gena Hahn (graph theory)
* Jean-François Gauthier (learning algorithms in general)

Parts of this project were made using ChatGPT.
