# Winner-Take-All Network Simulation

## Program Overview

### Usage Instructions:
This script simulates a Winner-Take-All (WTA) network using PyNN with SpiNNaker. It creates a network of layers with neurons connected by lateral inhibition and recurrent excitation, processes spike data, and visualizes the results.

- **Number of Layers and Neurons per Layer**: Define the network structure, where `num_layers` specifies the number of layers, and `neurons_per_layer` specifies how many neurons are in each layer.

- **Strengths of Connections**: The strengths for lateral inhibition and recurrent excitation are manually set to control their impact.

- **Stimulus Strength**: The strength of the input stimulus is set to a low value to prevent saturation.

- **Recording and Visualization**: The simulation records spike data from each neuron and generates a scatter plot of spikes over time.

### Example Plot:
https://imgur.com/a/X5HOPbz
(This image is a placeholder; actual results will vary based on simulation data.)

## Winner-Take-All Network Algorithm

### Algorithm Overview
This algorithm implements a Winner-Take-All mechanism in a network of spiking neurons to perform clustering. It utilizes a single-layer neural network with competitive learning, also known as the Kohonen learning rule.

- **Network Structure**: The network consists of multiple layers, with each neuron competing to represent an input stimulus.

### Algorithm Steps:

#### Initialization Phase:
1. Set up the simulation environment with a timestep of 1 millisecond using `p.setup(timestep=1.0)`.
2. Define the network size with `num_layers` and `neurons_per_layer`.
3. Create the input layer with Poisson spike sources and multiple hidden layers with leaky integrate-and-fire (IF_cond_exp) neurons.

#### Connectivity Phase:
1. **Lateral Inhibition**: Neurons within the same layer inhibit each other to enhance competition. This is implemented with a fixed probability of connection and a low weight (0.01).
2. **Recurrent Excitation**: Neurons receive excitation from themselves and their neighbors to strengthen their activation. This is implemented with a fixed probability of connection and a very low weight (0.005).
3. **Topographic Connections**: Connect the input layer to each hidden layer with excitatory connections, using a fixed probability and a low stimulus strength (0.05) to avoid saturation.

#### Simulation Phase:
1. Record spike activity from all neurons.
2. Run the simulation for 1000 milliseconds to allow the network to develop its response to inputs.

#### Data Retrieval and Visualization:
1. Extract and process spike train data from each layer.
2. Plot the spike times of all neurons across all layers in a scatter plot to visualize the activity of the network.

### Mathematical Details:
1. **Lateral Inhibition**:
   - Each neuron inhibits its neighbors to create a competitive environment. The inhibition strength is set using:
     \[
     \text{weight} = 0.01
     \]
   - Connectivity is probabilistic, with a connection probability of 0.1.

2. **Recurrent Excitation**:
   - Neurons receive excitatory input from themselves and their neighbors to reinforce activation. The excitation strength is set using:
     \[
     \text{weight} = 0.005
     \]
   - Connections are established with a 0.1 probability.

3. **Stimulus Strength**:
   - Input to neurons is scaled by a strength factor to avoid overwhelming the network. The strength is set as:
     \[
     \text{weight} = 0.05
     \]
   - This connects the input layer to each hidden layer with a fixed probability of 0.1.

### Characteristics Analysis:
1. **Impact of Connection Strengths**:
   - Adjusting the strengths of lateral inhibition and recurrent excitation affects the clustering result and network dynamics. Lower strengths reduce the impact of these connections, potentially leading to less pronounced clustering.

2. **Effect of Initialization and Connectivity**:
   - Proper initialization and connectivity are crucial for effective clustering. Improper initialization or extreme connectivity parameters may result in imbalanced clustering or ineffective competition among neurons.

3. **Network Size and Clustering**:
   - The number of layers and neurons per layer influences the network's ability to differentiate between clusters. Incorrect settings may lead to inadequate clustering or failure to capture the underlying data structure.

### Conclusion:
This code provides a practical implementation of a Winner-Take-All network using a spiking neural network approach. It emphasizes the fundamentals of competitive learning and clustering in a neural network context. The focus is on understanding and applying the basic principles of unsupervised learning rather than achieving optimal clustering performance.
