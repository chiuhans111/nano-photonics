import numpy as np
import matplotlib.pyplot as plt

with open("./MTF/Phase_mask(Ideal_Amp_Phase).dat") as f:
    lines = f.readlines()
    data = []
    x_range = [float(v) for v in lines[2].split(" ")[:3]]
    y_range = [float(v) for v in lines[3].split(" ")[:3]]
    for line in lines[4:]:
        value = [v.strip() for v in line.split(' ')]
        data.append([float(v) for v in value if len(v) > 0])


def phase_average(value, weight=1):
    n = np.exp(value*np.pi*2j)*weight
    return np.angle(np.mean(n)/np.mean(weight))/np.pi/2


data = np.array(data)

print("\n"*5)
print(x_range)
print(y_range)
X = np.linspace(*x_range[1:], int(x_range[0]))
Y = np.linspace(*x_range[1:], int(y_range[0]))
X, Y = np.meshgrid(X, Y)

extent = [*x_range[1:], *y_range[1:]]

amplitude = data[:, ::2]**2
phase = data[:, 1::2] / 360

focal_length = 12
wavelength = 0.55


# ideal_lens
distance = np.sqrt(X**2+Y**2+focal_length**2)-focal_length
phase_ideal = distance/wavelength

diff = (phase - phase_ideal) % 1
diff -= phase_average(diff, amplitude)
diff = (diff+0.5) % 1-0.5

phase_fixed = phase_ideal + diff
plt.imshow(phase_fixed, extent=extent)
plt.show()

plt.hist(diff.flatten(), bins=100)
plt.show()


def write_to_int(filename, phase, wavelength, ssz=2048):
    phase -= np.mean(phase)
    with open(filename, 'w') as f:
        f.write("FIX METALENS\n")
        f.write(f"GRD {phase.shape[0]} {phase.shape[1]} ")
        f.write(f"WFR WVL {wavelength} SSZ {ssz}\n")
        c = 0
        for i in phase.flatten():
            f.write(f"{round(i*ssz)} ")
            c += 1
            if c % 10 == 0:
                f.write("\n")


write_to_int('./MTF/fix.int', phase_fixed, wavelength)
write_to_int('./MTF/unfix.int', phase, wavelength)
write_to_int('./MTF/ideal.int', phase_ideal, wavelength)
