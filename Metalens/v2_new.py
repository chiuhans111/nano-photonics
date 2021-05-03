import numpy as np
import matplotlib.pyplot as plt

from most_read import get_variable, get_result
from utils import angle_average, angle_normalize
from genetic import GeneticOptimizer

variables = get_variable("./single/search_range.txt")
results = get_result("./single/mosttmp_work")["mosttmp_dm_m3_ex"]

W = variables["W"]
H = variables["H"]

# PERIOD = variables["PERIOD"]
A = results[:, 0]
P = results[:, 1]


# FIND AND PLACE AND BUILD----------------------------------------------

wavelength = 1
size = 501
period = 0.6
fl = 3*1000

X_axis = np.linspace(0, 1, size) * (size-1) * period


# up sample by 10
# X_axis_more = np.linspace(-1, 1, size*10) / 2 * (size-1)*period
# distance_more = np.sqrt(fl**2+X_axis_more**2)
# phase_distance_more = distance_more / wavelength
# directivity_more = fl/distance_more


class GO(GeneticOptimizer):
    def populate(self):
        return np.random.randint(0, len(W), [self.NP, size])

    def mix(self, a, b):
        cross_over = np.random.rand(len(a)) < 0.5
        return a + cross_over*(b-a)

    def mutate(self, a):
        a = np.copy(a)
        axis = np.random.randint(0, size)
        a[axis] = np.random.randint(0, len(W))
        return a

    def fitness(self, selected):
        P_gets = P[selected]
        A_gets = A[selected]

        goal = 0

        distance = np.sqrt(fl**2+X_axis**2)
        phase_distance = distance / wavelength
        directivity = fl/distance

        U = np.exp((P_gets/360) * 2j*np.pi) * A_gets

        U = np.interp(X_axis, X_axis, U)
        U_trans = U * \
            np.exp(phase_distance * 2j*np.pi) / \
            distance*directivity

        intensity = np.abs(np.mean(U_trans))
        goal += intensity

        return goal


go = GO(30, 0.9)
iterations = 5000

for i, selected in go.optimize(iterations):

    P_gets = P[selected]
    A_gets = A[selected]

    if i % 100 == 99:
        print(i)
        plt.clf()
        # plt.plot(X_axis, (P_wants-P_wants[0]) % 360, color='blue')
        plt.scatter(X_axis, (P_gets-P_gets[0]) % 360, color='blue')
        plt.twinx()
        # plt.plot(X_axis, A_wants/np.mean(A_wants), color='red')
        plt.scatter(X_axis, A_gets/np.mean(A_gets), color='red')
        plt.pause(0.01)

plt.show()
# Build and save to file -----------------------------------------

selected = list(selected)
selected = selected[::-1] + selected[1:]

selectedWH = np.array([
    W[selected],
    H[selected]
]).T

plt.scatter(W, H, A)
plt.plot(*selectedWH.T)
plt.xlabel("W")
plt.ylabel("H")
plt.show()

with open("./template/multiplerect.ind", 'r') as f:
    text = f.read().format(
        SIZE=len(selectedWH)
    )

with open("./template/channel_segment.ind", "r") as f:
    segment_template = f.read()

segments = [[0, 0, 0, "width", "height", 0.1]]

for i in range(len(selectedWH)):
    x = f"{i+0.5}*PERIOD-width/2"
    y = 0
    z = 0.1
    w = f"{selectedWH[i][0]} * PERIOD"
    h = f"{selectedWH[i][1]} * PERIOD"
    segments.append([x, y, z, w, h, "TALL"])


def make_segment(id, x, y, z, w, h, tall):
    return segment_template.format(
        id=id, x=x, y=y, z=z, w=w, h=h, tall=tall
    )


for i, segment in enumerate(segments):
    text += "\n\n"+make_segment(i+1, *segment)

with open("./multiple/multiplerect.ind", 'w') as f:
    f.write(text)
