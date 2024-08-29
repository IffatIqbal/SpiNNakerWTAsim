import pyNN.spiNNaker as p
import numpy as np
import matplotlib.pyplot as plt

# Initialize simulation environment
p.setup(timestep=1.0)

# Define the network size and structure
num_layers = 6       # Number of layers in the network
neurons_per_layer = 100  # Number of neurons in each layer

# Create populations for input and layers
input_layer = p.Population(neurons_per_layer, p.SpikeSourcePoisson(rate=10), label="Input Layer")
layers = []
for i in range(num_layers):
    layer = p.Population(neurons_per_layer, p.IF_cond_exp(), label=f"Layer {i+1}")
    layers.append(layer)

# Reduced inhibitory lateral connections and recurrent excitatory connections to avoid saturation
lateral_inhibition_strength = 0.01  # Reduced strength of lateral inhibition
recurrent_excitation_strength = 0.005  # Reduced strength of recurrent excitation

# Connectivity function for lateral inhibition (connect neighbors only)
def lateral_inhibition():
    return p.FixedProbabilityConnector(p_connect=0.1), p.StaticSynapse(weight=lateral_inhibition_strength)

# Connectivity function for recurrent excitation (self and neighbors)
def recurrent_excitation():
    return p.OneToOneConnector(), p.StaticSynapse(weight=recurrent_excitation_strength)

# Connect each layer to itself with lateral inhibition and recurrent excitation
for layer in layers:
    p.Projection(layer, layer, lateral_inhibition()[0], synapse_type=lateral_inhibition()[1], receptor_type='inhibitory')
    p.Projection(layer, layer, recurrent_excitation()[0], synapse_type=recurrent_excitation()[1], receptor_type='excitatory')

# Topographic connections from input to layers
stimulus_strength = 0.05  # Reduced input stimulus strength to avoid saturation
input_to_layers = p.FixedProbabilityConnector(p_connect=0.1)
for layer in layers:
    p.Projection(input_layer, layer, input_to_layers, synapse_type=p.StaticSynapse(weight=stimulus_strength), receptor_type='excitatory')

# Record spikes for visualization
for layer in layers:
    layer.record(['spikes'])

# Run simulation for a given time
simulation_time = 1000  # Time in milliseconds
p.run(simulation_time)

# Retrieve and process spike data
spikes_data = []
for i, layer in enumerate(layers):
    # Convert Block object to spiketrain list
    spike_trains = layer.get_data('spikes').segments[0].spiketrains
    spikes_data.append(spike_trains)

# Plot all spikes from t=0 to t=final in one image
plt.figure(figsize=(10, 8))

for i, spike_trains in enumerate(spikes_data):
    for train in spike_trains:
        train_times = np.array(train)  # Get spike times
        plt.scatter(train_times, np.ones_like(train_times) * (i * neurons_per_layer + train.annotations['source_index']), s=2)

plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Winner-Take-All Network Spikes from t=0 to t=final')
plt.savefig('wta_simulation_full.png')
plt.show()

# End the simulation
p.end()
