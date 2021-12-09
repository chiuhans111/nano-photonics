import enum
import numpy as np
import matplotlib.pyplot as plt

# IFTA algorithms


def get_factor(S_begin):
    s = len(S_begin)
    factor = np.sqrt(np.mean(np.abs(S_begin)**2)/s)
    return factor


def IFTA_basic(S2):
    factor = get_factor(S_begin)

    for i in range(1000):

        U = np.fft.ifft(S2)
        U2 = np.exp(1j*np.angle(U))
        S = np.fft.fft(U2)*factor
        S2 = np.exp(1j*np.angle(S))*T

        loss = np.mean((np.abs(S)**2-T)**2)

        yield U2, S, loss


def IFTA_momenta(S2, ratio=0.1, epsilon=0.1):
    factor = get_factor(S_begin)

    momenta = 0
    U = 0

    for i in range(1000):

        U_last = U

        U = np.fft.ifft(S2)

        if i > 0:
            U = U / np.mean(U)
            delta = (U - U_last)
            delta = delta / (np.abs(delta)+epsilon)
            momenta += delta
            U = momenta + U_last
            momenta *= ratio

        U2 = np.exp(1j*np.angle(U))

        S = np.fft.fft(U2)*factor

        S2 = np.exp(1j*np.angle(S))*T

        loss = np.mean((np.abs(S)**2-T)**2)

        yield U2, S, loss


def IFTA_angle_momenta(S2, ratio=0.5, epsilon=0.1):
    factor = get_factor(S_begin)

    momenta = 0
    speed = 1
    U = 0
    for i in range(1000):

        U_last = U

        U = np.fft.ifft(S2)

        if i > 0:

            delta = (np.angle(U) - np.angle(U_last))
            delta = delta - np.angle(np.sum(np.exp(1j*delta)))
            delta = (delta + np.pi) % (2*np.pi)-np.pi
            speed = 0.2/(np.mean((momenta - delta)**2)+epsilon)
            momenta = delta * ratio  + momenta * (1-ratio)

            U = np.exp(1j*(delta * speed + np.angle(U_last)))

        U2 = np.exp(1j*np.angle(U))

        S = np.fft.fft(U2)*factor

        S2 = np.exp(1j*np.angle(S))*T

        loss = np.mean((np.abs(S)**2-T)**2)

        yield U2, S, loss


# basic coordinates
x = np.linspace(0, 1, 100)

T = (x % 0.5 < 0.2)*1
T = np.sqrt(T)
plt.plot(T)
plt.show()

while True:

    S_begin = T * np.exp(2j*np.pi*np.random.rand(len(T)))

    algorithms = []
    methods = []

    # APPEND ALGORITHMS
    methods.append("basic")
    algorithms.append(IFTA_basic(S_begin))
    
    methods.append("momenta")
    algorithms.append(IFTA_momenta(S_begin))

    for i in range(5):
        ratio = i/5+0.1
        methods.append(f"angular {ratio:.2f}")
        algorithms.append(IFTA_angle_momenta(S_begin, ratio))

    # START SIMULATION

    size = len(methods)

    histories = []
    for i in range(size):
        histories.append([])

    for iteration, results in enumerate(zip(*algorithms)):

        do_plot = False

        if do_plot:
            plt.clf()

        for id, result in enumerate(results):
            U2, S, loss = result

            histories[id].append(loss)

            name = methods[id]

            if do_plot:
                plt.subplot(3, 1, id+1)
                plt.title(name)

                plt.plot(np.real(U2))
                # plt.subplot(2, size, id+1 + size)

        if do_plot:
            # plt.subplot(2,1,2)
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.plot(histories[0], label="BASIC")
            # plt.plot(histories[1], label="MOMENTA")
            # plt.plot(histories[2], label="ANGLE MOMENTA")
            plt.pause(0.01)

    print("DONE")

    plt.show()

    for id, history in enumerate(histories):
        plt.plot(history, label=methods[id])

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()
