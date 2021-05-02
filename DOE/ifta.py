import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from numba import jit
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)



@tf.function
def picture2doe(my_picture, iter=100, ini_phase=None):

    my_picture = tf.cast(my_picture, tf.dtypes.float32)

    my_picture = my_picture / 255
    my_picture = tf.sqrt(my_picture)
    my_picture = tf.signal.fftshift(my_picture)
    # plt.imshow(my_picture)
    # plt.show()

    if ini_phase is None:
        ini_phase = tf.random.uniform(my_picture.shape) * 2 * np.pi
    ini_phase = tf.cast(ini_phase, tf.dtypes.complex64)

    my_picture = tf.cast(my_picture, tf.dtypes.complex64)
    Cg = my_picture * tf.math.exp(1j*ini_phase)

    g = tf.signal.ifft2d(Cg)
    g_factor = tf.sqrt(tf.reduce_mean(abs(g)**2))
    g_factor = tf.cast(g_factor, tf.dtypes.complex64)
    Niteration = iter

    steps = 256
    for n in range(Niteration):
        g = tf.math.divide_no_nan(
            g_factor * g, tf.cast(tf.abs(g), tf.dtypes.complex64))

        Cg = tf.signal.fft2d(g)
        # Cg_phase = tf.math.angle(Cg)
        # Cg_phase = tf.cast(Cg_phase, tf.dtypes.complex64)
        # Cg = my_picture * tf.math.exp(1j*Cg_phase)
        Cg = tf.math.divide_no_nan(
            Cg * my_picture,  tf.cast(tf.abs(Cg), tf.dtypes.complex64))

        g = tf.signal.ifft2d(Cg)
        # g_phase = tf.math.angle(g)
        # g_phase = tf.round(g_phase/np.pi/2*steps) % steps
        # g_phase = tf.cast(g_phase, tf.dtypes.complex64)
        # g = g_factor * tf.math.exp(1j*g_phase/steps*np.pi*2)


        # if n % 10 == 0:
        #     intCg = abs(Cg)**2
        #     plt.subplot(2, 5, n//10+1)
        #     plt.imshow(intCg)
        #     plt.title(f"n={n}")

        #     plt.pause(0.01)

    # doe = tf.cast(g_phase, tf.dtypes.float32)
    doe = tf.math.angle(g)/np.pi/2*steps % steps

    # doe = (doe - tf.reduce_mean(doe)) % 256
    tile1 = tf.cast(tf.math.ceil(1080/doe.shape[0]), tf.dtypes.int32)
    tile2 = tf.cast(tf.math.ceil(1920/doe.shape[1]), tf.dtypes.int32)
    doe = tf.tile(doe, [tile1, tile2])[:1080, :1920]

    return tf.cast(doe, tf.dtypes.int32)[::-1, ::-1], Cg, tf.math.angle(Cg)
