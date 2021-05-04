import numpy as np


def read(file_path, expand_sym=False):
    with open(file_path) as f:
        lines = f.readlines()
        result = []
        for line in lines[4:]:
            value = [v.strip() for v in line.split(' ')]
            result.append([float(v) for v in value if len(v) > 0])
    result = np.array(result)

    a = result[:, ::2]
    b = result[:, 1::2]

    if expand_sym:
        a = np.concatenate([a[::-1], a[1:]], 0)
        a = np.concatenate([a[:, ::-1], a[:, 1:]], 1)

        b = np.concatenate([b[::-1], b[1:]], 0)
        b = np.concatenate([b[:, ::-1], b[:, 1:]], 1)

    return a, b


def get(file_path, expand_sym=False):
    amplitude, phase = read(file_path, expand_sym)
    phase = phase/180*np.pi
    metalens_field = np.exp(-phase*1j) * amplitude
    return metalens_field


def get_im(file_path, expand_sym=False):
    real, imag = read(file_path, expand_sym)
    return real+1j*imag


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    metalens_field = get("./MTF/Real_Phase_Mask_NA_0p3_m3_f2_ey.dat")
    phase = np.angle(metalens_field)*180/np.pi

    x = np.linspace(-1, 1, 864) * 6.4725
    wavelength = 0.5
    focal_length = 11.6
    distance = np.sqrt(x**2+focal_length**2) - focal_length
    phase2 = (distance/wavelength*360) % 360

    plt.plot(x, (phase[phase.shape[0]//2] -
                 phase[phase.shape[0]//2, phase.shape[1]//2]) % 360)
    plt.plot(x, phase2)
    plt.show()
