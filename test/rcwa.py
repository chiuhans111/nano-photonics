import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


