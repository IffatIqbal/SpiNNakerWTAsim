import pyNN.spiNNaker as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, simulation_time)
ax.set_ylim(0, num_layers * neurons_per_layer)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron Index')
ax.set_title('Winner-Take-All Network Spikes across Layers')

# Function to update each frame
def update(frame):
    ax.clear()
    ax.set_xlim(0, simulation_time)
    ax.set_ylim(0, num_layers * neurons_per_layer)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron Index')
    ax.set_title(f'Winner-Take-All Network Spikes (Time {frame}-{frame+50}ms)')
    
    for i, spike_trains in enumerate(spikes_data):
        for train in spike_trains:
            train_times = np.array(train)
            spikes_in_frame = train_times[(train_times >= frame) & (train_times < frame + 50)]
            ax.scatter(spikes_in_frame, np.ones_like(spikes_in_frame) * (i * neurons_per_layer + train.annotations['source_index']), s=2)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(0, simulation_time, 50), interval=200)

# Save the animation as a GIF using PillowWriter
ani.save('wta_simulation.gif', writer=animation.PillowWriter(fps=5))


# Save a static image from the first frame (e.g., t=0 to t=50ms)
update(0)
plt.savefig('wta_simulation_static.png')

# End the simulation
p.end()
