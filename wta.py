
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
from matplotlib.animation import FuncAnimation, PillowWriter

# Setup the SpiNNaker simulation
p.setup(timestep=1.0)

# Define parameters
n_exc_neurons = 100  # Number of excitatory neurons
n_inh_neurons = 1    # Number of inhibitory interneurons
w_exc = 0.1          # Weight of excitatory connections
w_inh = -1.0         # Weight of inhibitory connections
delay = 2.0          # Delay for inhibitory feedback

# Create populations of neurons
exc_neurons = p.Population(n_exc_neurons, p.IF_curr_exp(), label="Excitatory Neurons")
inh_neurons = p.Population(n_inh_neurons, p.IF_curr_exp(), label="Inhibitory Neuron")

# Create connections
# Excitatory to inhibitory connection (all excitatory neurons connect to the inhibitory neuron)
exc_to_inh = p.Projection(exc_neurons, inh_neurons, p.AllToAllConnector(), p.StaticSynapse(weight=w_exc))

# Inhibitory feedback (inhibitory neuron connects back to all excitatory neurons)
inh_to_exc = p.Projection(inh_neurons, exc_neurons, p.AllToAllConnector(), p.StaticSynapse(weight=w_inh, delay=delay))
