# in-silico-BNN
This project aims to emulate the behavior of organoid intelligence in a numerical environnement. It also aims to compute various definitions of the system’s free energy to analyze how they evolve over time and investigate their correlation with the model’s learning dynamics.

## Getting started

### Prerequisite
* Python 3.10
* GIT

### Installation
To create a local copy of this repository, use the ```git clone``` command with the project's URL:

 ```
 git clone https://github.com/lashlone/in-silico-BNN.git
 ```

Next, install the required Python libraries listed in the [requirements](requirements.txt) file:

 ```
 pip install -r requirements.txt
 ```

I strongly recommend creating a virtual environnement to ensure package version compatibilities and to isolate dependencies from your system Python environnement. You can use any tool you are comfortable with (e.g., ```virtualenv```, ```conda```). Here's how to do it using Python’s built-in ```venv``` module, which doesn’t require any additional setup:

 ```
 python -m venv <environnement_name>
 source <environment_name>/bin/activate  # On Linux/macOS
 <environment_name>\Scripts\activate     # On Windows
 ```

### Create a network

The [network](\network) module contains all the functions and classes to create, use and analyze a neural network based on my model. To create a Network object, you must first define its regions. Regions objects are initialized by their size, name and type. The size of a region corresponds to the number of neurons within them, while their name later allows to identify them in the network. In this preliminary version, region types only influence the math behind the system's free energy computation. However, I plan to implement more region's type, such as inhibitory or non-spiking neurons, to allow more complexity in the model.

### Run a simulation

The [simulation](\network) module

### Analyze your data

All of the [...]

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

This project was made using ChatGPT.
