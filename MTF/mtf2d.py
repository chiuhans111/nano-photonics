if __name__ == "__main__":
    import sys
    sys.path.append('./')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import phasemask
from matplotlib import cm
from scipy import interpolate
for gpu in tf.config.list_physical_devices('GPU'):
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, False)


@tf.function
def propagation_integration(from_coord, field, to_point, wavelength):
    from_coord = tf.cast(from_coord, tf.dtypes.float32)
    field = tf.cast(field, tf.dtypes.complex64)
    to_point = tf.cast(to_point, tf.dtypes.float32)

    displace = to_point - from_coord
    distance = tf.sqrt(tf.reduce_sum(displace**2, -1))
    phase = distance / wavelength * 2 * np.pi
    directivity = displace[:, :, 2]/distance,

    phase = tf.cast(phase, tf.dtypes.complex64)
    distance = tf.cast(distance, tf.dtypes.complex64)
    directivity = tf.cast(directivity, tf.dtypes.complex64)

    phasor = tf.math.exp(1j * phase)
    return tf.reduce_sum(phasor * field / distance * directivity)


def propagation(from_coord, field, to_coord, wavelength):
    from_coord = tf.cast(from_coord, tf.dtypes.float32)
    field = tf.cast(field, tf.dtypes.complex64)
    to_coord = tf.cast(to_coord, tf.dtypes.float32)

    to_shape = to_coord.shape
    target_shape = [-1] + [1]*(len(from_coord.shape)-1) + [to_shape[-1]]
    to_coord_flatten = tf.reshape(to_coord, target_shape)
    result_shape = to_shape[:-1]

    result = []
    size = to_coord_flatten.shape[0]
    for i in range(size):
        if i % 1000 == 0:
            tf.print(f"{i/size*100:.2f}%")

        point_integral = propagation_integration(
            from_coord, field, to_coord_flatten[i], wavelength)
        result.append(point_integral)

    result = tf.cast(result, tf.dtypes.complex64)
    result = tf.reshape(result, result_shape)
    return result


def grid(width, height, z, resolution_X, resolution_Y):
    X = tf.cast(tf.linspace(-1, 1, resolution_X) * width, tf.dtypes.float32)
    Y = tf.cast(tf.linspace(-1, 1, resolution_Y) * height, tf.dtypes.float32)
    X, Y = tf.meshgrid(X, Y)
    Z = tf.ones([resolution_Y, resolution_X], tf.dtypes.float32) * z
    return tf.stack([X, Y, Z], 2)


def gridse(xs, xe, ys, ye, z, resolution_X, resolution_Y):
    X = tf.cast(tf.linspace(xs, xe, resolution_X), tf.dtypes.float32)
    Y = tf.cast(tf.linspace(ys, ye, resolution_Y), tf.dtypes.float32)
    X, Y = tf.meshgrid(X, Y)
    Z = tf.ones([resolution_Y, resolution_X], tf.dtypes.float32) * z
    return tf.stack([X, Y, Z], 2)


def vgrid(width, height, resolution_X, resolution_Y):
    X = tf.cast(tf.linspace(-1, 1, resolution_X) * width, tf.dtypes.float32)
    Z = tf.cast(tf.linspace(0, 1, resolution_Y) * height, tf.dtypes.float32)
    X, Z = tf.meshgrid(X, Z)
    Y = tf.zeros([resolution_Y, resolution_X], tf.dtypes.float32)
    return tf.stack([X, Y, Z], 2)


def ideal_lens(coord, target_point, wavelength):
    coord = tf.cast(coord, tf.dtypes.float32)
    target_point = tf.cast(target_point, tf.dtypes.float32)

    displace = coord - target_point
    distance = tf.sqrt(tf.reduce_sum(displace**2, -1))
    phase = -distance / wavelength * 2 * np.pi

    phase = tf.cast(phase, tf.dtypes.complex64)

    phasor = tf.math.exp(1j * phase)
    return phasor


def aperture(coord, radius):
    distance = tf.reduce_sum(coord**2, -1)
    return tf.cast(distance < radius**2, tf.dtypes.complex64)


def circle(x, y, radius):
    X = np.linspace()


# BASIC PARAMETERS
# -----------------------------------------------------------------------------

resolution = 701
mon_resolution = 64

lens_radius = 10.5
focal_length = 10
mon_radius = 10
wavelength = 0.55


diagonal = np.sqrt(lens_radius**2+focal_length**2)
numerical_aperture = lens_radius / diagonal
airy_radius = 1.22 * wavelength * diagonal / lens_radius / 2
cutoff_frequency = 2 * numerical_aperture / wavelength

print("\n"*5)
print("INFORMATIONS-----------------")
print("NA:", numerical_aperture)
print("airy radius:", airy_radius)
print("cutoff frequency:", cutoff_frequency)
print("\n"*5)


in_coord = grid(lens_radius,  lens_radius, 0, resolution, resolution)

to_coord = grid(mon_radius, mon_radius, focal_length,
                mon_resolution, mon_resolution)

xz_coord = vgrid(mon_radius, focal_length*1.5, mon_resolution, mon_resolution)


ideallens_field = ideal_lens(in_coord, [[[0, 0, focal_length]]], wavelength)
ideallens_field = ideallens_field * aperture(in_coord, lens_radius)


metalens_field = phasemask.get("./fwtmp_m443_f2_ex.dat")
metalens_field = tf.cast(metalens_field, tf.dtypes.complex64)

plt.imshow(np.angle(metalens_field))
plt.show()
# START SIMULATE


lens_field = metalens_field
lens_name = "meta_me"


xz_result = propagation(in_coord, lens_field, xz_coord, wavelength)

plt.figure(f"{lens_name}_prop")
plt.title("propagation")
plt.pcolor(xz_coord[:, :, 0], xz_coord[:, :, 2], tf.abs(xz_result))
plt.axis('equal')

plt.gca().add_artist(plt.Circle((0, focal_length),
                                airy_radius, fill=None, color='r'))
plt.ylabel("Z axis ($\mu m$)")
plt.xlabel("X axis ($\mu m$)")
plt.tight_layout()
plt.show()


result = propagation(in_coord, lens_field, to_coord, wavelength)

# plt.pcolor(in_coord[:, :, 1], in_coord[:, :, 0], tf.math.angle(lens_field))
# plt.show()

plt.figure(f"{lens_name}_PSF")
plt.title("PSF")
plt.pcolor(to_coord[:, :, 0], to_coord[:, :, 1], tf.abs(result))
plt.axis('equal')

plt.gca().add_artist(plt.Circle((0, 0), airy_radius, fill=None, color='r'))
plt.ylabel("Y axis ($\mu m$)")
plt.xlabel("X axis ($\mu m$)")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.set_window_title(f"{lens_name}_PSF3D")
ax.set_title("PSF 3D")
surf = ax.plot_surface(to_coord[:, :, 0], to_coord[:, :, 1],
                       tf.abs(result), cmap=cm.coolwarm)
plt.tight_layout()
plt.show()

psf = tf.cast(tf.abs(result)**2, tf.dtypes.complex64)
psf = psf / tf.reduce_sum(psf)

psfint = interpolate.interp2d(
    to_coord[0, :, 0], to_coord[:, 0, 1], tf.abs(psf), kind='cubic')


plt.figure(f"{lens_name}_PSF_slice")

r = np.linspace(-mon_radius, mon_radius, mon_resolution*2)

for angle in [0, 10, 30, 45, 60, 90]:
    x = np.cos(angle/180*np.pi) * r
    y = np.sin(angle/180*np.pi) * r
    psf_section = []
    for i in range(r.shape[0]):
        psf_section.append(psfint(x[i], y[i]))
    plt.plot(r, psf_section, label=f"angle = {angle} deg")

plt.grid()
plt.legend()
plt.title("PSF")
plt.ylabel("PSF")
plt.xlabel("displacement ($\mu m$)")
plt.tight_layout()
plt.show()


mtf = tf.signal.fftshift(tf.signal.fft2d(psf))

fx = np.fft.fftfreq(mtf.shape[1], d=to_coord[0, 1, 0]-to_coord[0, 0, 0])
fy = np.fft.fftfreq(mtf.shape[0], d=to_coord[1, 0, 1]-to_coord[0, 0, 1])
fx, fy = np.meshgrid(fx, fy)
fx = np.fft.fftshift(fx)
fy = np.fft.fftshift(fy)

plt.figure(f"{lens_name}_MTF")
plt.title("MTF")
plt.pcolor(fx, fy, tf.abs(mtf))
plt.gca().add_artist(plt.Circle((0, 0), cutoff_frequency, fill=None, color='r'))
plt.axis("equal")
plt.ylabel("Y frequency (cycles/$\mu m$)")
plt.xlabel("X frequency (cycles/$\mu m$)")
plt.xlim([-cutoff_frequency, cutoff_frequency])
plt.ylim([-cutoff_frequency, cutoff_frequency])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.set_window_title(f"{lens_name}_MTF3D")
ax.set_title("MTF 3D")
surf = ax.plot_surface(fx, fy, tf.abs(mtf), cmap=cm.coolwarm)

plt.tight_layout()
plt.show()


mtfint = interpolate.interp2d(fx[0, :], fy[:, 0], tf.abs(mtf), kind='cubic')

plt.figure(f"{lens_name}_MTF_slice")
plt.plot([0, cutoff_frequency], [1, 0], '--',
         c='k', alpha=0.5, label='limit')
f = np.linspace(0, cutoff_frequency, 100)

for angle in [0, 10, 30, 45, 60, 90]:
    x = np.cos(angle/180*np.pi) * f
    y = np.sin(angle/180*np.pi) * f
    mtf_section = []
    for i in range(f.shape[0]):
        mtf_section.append(mtfint(x[i], y[i]))
    plt.plot(f, mtf_section, label=f"angle = {angle} deg")

plt.grid()
plt.legend()
plt.title("MTF")
plt.ylabel("MTF")
plt.xlabel("frequency (cycles/$\mu m$)")
plt.tight_layout()
plt.show()
