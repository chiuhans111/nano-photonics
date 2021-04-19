import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraypad import _pad_simple

# basic constants
diameter = 20
lens_diameter = 15
fl = 20
wavelength = 0.5
size = 400
dx = diameter/size

if wavelength/dx < 4:
    print("warning! sample too low")

# coordinate
X = np.linspace(-1, 1, size) * diameter / 2


def perfect_lens(X, fl, phase_clip=0, anoise=0):
    d = np.sqrt(X**2+fl**2)
    phase = d/wavelength % 1
    return np.exp(-phase*2j*np.pi)


field1 = perfect_lens(X, fl, 0, 0)
field2 = field1
field3 = field1 * np.random.rand(field1.shape[0])


field1 = field1*(np.abs(X) < lens_diameter/2)
field2 = field2*(np.abs(X) < lens_diameter/2)
field3 = field3*(np.abs(X) < lens_diameter/2)

plt.figure()
plt.plot(np.angle(field1))
plt.plot(np.angle(field2))
plt.plot(np.angle(field3))
plt.show(block=False)


def propagation(X, field, z):

    if z == 0:
        return field
    else:
        X1 = X[:, np.newaxis]
        X2 = X[np.newaxis, :]
        d = np.sqrt((X1-X2)**2+z**2)
        phase = d*2j*np.pi/wavelength
        directivity = z/d
        area = np.abs(X)
    return np.sum(field*np.exp(phase)*directivity, 1)


# fields = []
# z_near = 0
# z_far = fl*3
# for z in np.linspace(z_near, z_far, 100):
#     fields.append(propagation(X, field, z))
# fields = np.array(fields)

# plt.imshow(np.abs(fields)**2,
#            extent=[-diameter/2, diameter/2, z_far, z_near])
# plt.show()

lsf1 = propagation(X, field1, fl)
lsf2 = propagation(X, field2, fl)
lsf3 = propagation(X, field3, fl)
plt.figure()
plt.plot(np.abs(lsf1/np.max(lsf1))**2)
plt.plot(np.abs(lsf2/np.max(lsf2))**2)
plt.plot(np.abs(lsf3/np.max(lsf3))**2)
plt.show(block=False)


def MTF(field):
    full_size = 2
    while full_size < field.shape[0]*4:
        full_size *= 2
    pad_size = (full_size-field.shape[0])//2
    field = np.pad(np.abs(field)**2, ([pad_size, pad_size]))
    field /= np.sum(field)
    mtf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))
    freq = np.fft.fftshift(np.fft.fftfreq(mtf.shape[0], dx))
    return freq, mtf


cutoff = 1 / wavelength * lens_diameter / fl

f, mtf1 = MTF(lsf1)
f, mtf2 = MTF(lsf2)
f, mtf3 = MTF(lsf3)
plt.figure()
plt.plot(f, np.abs(mtf1))
plt.plot(f, np.abs(mtf2))
plt.plot(f, np.abs(mtf3))
plt.plot([0, cutoff], [1, 0], "--")
plt.xlim([0, cutoff])
plt.show()
