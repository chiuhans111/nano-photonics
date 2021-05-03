import numpy as np
import matplotlib.pyplot as plt

from most_read import get_variable, get_result
from slabwaveguide import neff_of_channel
from utils import angle_average


variables = get_variable("./single/search_range.txt")
results = get_result("./single/mosttmp_work")["mosttmp_dm_m3_ex"]

print(variables)
print(results.shape)

W = variables["W"]
H = variables["H"]

# PERIOD = variables["PERIOD"]
A = results[:, 0]
P = results[:, 1]

if "TALL" in variables:
    TALL = variables["TALL"]
    TALL_set = np.sort(np.unique(TALL))

    max_gaps = []

    for tall in TALL_set:
        mask = TALL == tall
        p = np.sort(P[mask])
        print(p)
        p2 = np.roll(p, -1)
        max_gap = np.max((p2-p) % 360)
        max_gaps.append(max_gap)

    plt.scatter(TALL, P, A, "red")
    plt.plot(TALL_set, max_gaps)
    plt.ylabel("phase distribution")
    plt.xlabel("height")
    plt.show()

plt.xlabel("W")
plt.ylabel("H")
plt.scatter(W, H, s=50, c=A)
plt.quiver(W, H, np.cos(P/180*np.pi), np.sin(P/180*np.pi))
plt.show()


# Effective Index Approximate --------------------------------------------

neffs = []
for i in range(W.shape[0]):
    w = W[i] * 0.6
    h = H[i] * 0.6
    neffs.append(neff_of_channel(w, h, 1, 1.47, 1, 0))

neffs = np.array(neffs)
P_assume = (neffs*3*1*360) % 360
P_diff = P_assume-P
P_diff_ave = angle_average(P_diff)

plt.scatter(W, P_assume)
plt.scatter(W, P)
plt.show()


# neffs = W*H*1.47 + (1-W*H)*1

plt.xlabel("Neff")
plt.ylabel("P")
plt.scatter(neffs, (P_assume-P+180-P_diff_ave) % 360-180+P_diff_ave, A)
plt.axhline(P_diff_ave, color="red")
plt.show()

# FIND AND PLACE AND BUILD----------------------------------------------
size = 21
period = 0.6
fl = 10

X_axis = np.linspace(-1, 1, size) / 2 * (size-1)*period

P_wants = -np.sqrt(fl**2+X_axis**2)*360
# P_wants = np.linspace(0,360,size)*5

P_average = angle_average(P, A)
P_wants = (P_wants - angle_average(P_wants) + P_average) % 360

A_mean = np.mean(A)
A_max = np.max(A)

# P = P_assume

for iterate in range(100):
    angle_limit = 20-15*np.clip(iterate/90, 0, 1)

    selected = []
    P_gets = []
    A_gets = []
    A_diff = ((A-A_mean)/A_max)**2

    for P_want in P_wants:
        P_diff = (P-P_want+360+180) % 360-180
        fitness = A_diff + (P_diff/360)**2
        P_take = np.where(np.abs(P_diff) < angle_limit)[0]

        if len(P_take) > 0:
            best = P_take[np.argmin(fitness[P_take])]
        else:
            best = np.argmin(fitness)

        # print("target", P_want, "actuall", P[best])
        P_gets.append(P[best])
        A_gets.append(A[best])

        w = W[best]
        h = H[best]

        # plt.scatter(W[best], P_fit[best])
        # plt.axvline(w)
        # plt.show()
        w = np.clip(w, 0.05, 1)
        h = np.clip(h, 0.05, 1)
        selected.append([w, h])

    A_mean = np.mean(A_gets)

    # plt.clf()
    # plt.title(iterate)
    # plt.scatter(W, H)
    # plt.scatter(*np.array(selected).T, c='r')
    # plt.pause(0.05)
P_gets = np.array(P_gets)
plt.show()

plt.scatter(P_wants % 360, P_gets % 360)
plt.plot([0, 360], [0, 360], c="r")
plt.twinx()
plt.scatter(P_wants % 360, A_gets, color='red')
plt.show()

# Build and save to file -----------------------------------------

selected = np.array(selected)

plt.scatter(W, H, A)
plt.plot(*selected.T)
plt.xlabel("W")
plt.ylabel("H")
plt.show()

with open("./template/multiplerect.ind", 'r') as f:
    text = f.read().format(
        SIZE=len(selected)
    )

with open("./template/channel_segment.ind", "r") as f:
    segment_template = f.read()

segments = [[0, 0, 0, "width", "height", 0.1]]

for i in range(len(selected)):
    x = f"{i+0.5}*PERIOD-width/2"
    y = 0
    z = 0.1
    w = f"{selected[i][0]} * PERIOD"
    h = f"{selected[i][1]} * PERIOD"
    segments.append([x, y, z, w, h, "TALL"])


def make_segment(id, x, y, z, w, h, tall):
    return segment_template.format(
        id=id, x=x, y=y, z=z, w=w, h=h, tall=tall
    )


for i, segment in enumerate(segments):
    text += "\n\n"+make_segment(i+1, *segment)

with open("./multiple/multiplerect.ind", 'w') as f:
    f.write(text)
