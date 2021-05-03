import numpy as np
import matplotlib.pyplot as plt
from most_read import get_variable, get_result
from utils import angle_average
import tensorflow as tf

TRAIN_BASE_MODEL = False


# MOST RESULT ------------------------------------------------
variables = get_variable("./single/search_range.txt")
results = get_result("./single/mosttmp_work")["mosttmp_dm_m2_ex"]

print(variables)
print(results.shape)

W = variables["W"]
H = variables["H"]

# PERIOD = variables["PERIOD"]
er = results[:, 0]
ei = results[:, 1]

if False:
    plt.scatter(W, H, c=np.angle(er+1j*ei))
    plt.show()


# BASE MODEL ------------------------------------------------

def get_model():
    input_layer = tf.keras.Input([2])
    x = input_layer
    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(32, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(8, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(2)(x)
    return tf.keras.Model(input_layer, x)


train_x = tf.stack([W, H], 1)
train_y = tf.stack([er, ei], 1)

model = get_model()
adam = tf.keras.optimizers.Adam(lr=0.002)
model.compile(adam, 'mse')


try:
    model.load_weights("./models/model.h5")
except:
    print("no model to load")

if TRAIN_BASE_MODEL:
    Ws = np.linspace(min(W), max(W), 100)
    Hs = np.linspace(min(H), max(H), 100)
    Ws, Hs = np.meshgrid(Ws, Hs)

    test_x = tf.stack([Ws.flatten(), Hs.flatten()], 1)

    for i in range(100):
        model.fit(train_x, train_y, batch_size=32, epochs=50, shuffle=True)
        model.save("./models/model.h5")
        model.save("./models/model_backup.h5")
        test_y = model(test_x)
        result = np.reshape(test_y, list(Ws.shape) + [2])
        plt.clf()
        plt.imshow(np.angle(result[::-1, :, 0]+1j*result[::-1, :, 1]),
                   extent=[min(W), max(W), min(H), max(H)])
        plt.pause(0.01)
    plt.show()


# FIND ----------------------------------------------
for layers in model.layers:
    layers.trainable = False

size = 21
period = 1
fl = 20
wl = 0.55

# coordinates
X_axis = np.linspace(-1, 1, size) / 2 * (size-1)*period
Y_axis = np.linspace(-1, 1, size) / 2 * (size-1)*period
X_axis, Y_axis = np.meshgrid(X_axis, Y_axis)

P_wants = (fl-np.sqrt(fl**2+X_axis**2+Y_axis**2))/wl*np.pi*2 % np.pi*2


P_targets = np.array(list(set(P_wants.flatten())))
P_neg_phase = np.exp(-1j*P_targets)

print(P_wants.shape, P_targets.shape)


def evaluate(selected):
    e = np.array(model(selected))
    err = (e[:, 0] + 1j*e[:, 1]) * P_neg_phase
    mean = np.mean(err)
    mag = np.abs(mean)
    return (np.mean(np.abs(err - mean)**2)+1)/(mag+1e3)/(mag+2)


NP = 50

population = list(np.random.rand(NP, P_targets.shape[0], 2)*0.6+0.2)


for iterations in range(1000):

    for i in range(NP):
        j = np.random.randint(0, NP-1)
        k = np.random.randint(0, NP-2)
        if j >= i:
            j += 1

        if k >= min(i, j):
            k += 1
        if k >= max(i, j):
            k += 1

        me = population[i]
        other1 = population[j]
        other2 = population[k]
        mask = np.random.rand(P_targets.shape[0], 2) < 0.2
        rnd = np.random.rand(P_targets.shape[0], 2)-0.5
        population.append(me+(other2-other1+rnd*0.005)*(0.75)*mask)

    for i in range(len(population)):
        population[i] = np.clip(population[i], 0.2, 0.8)

    score = []
    for selected in population:
        score.append(evaluate(selected))

    population2 = []
    order = np.argsort(score)
    for i in range(NP):
        population2.append(population[order[i]])

    population = population2

    if iterations % 100 == 0:
        print(iterations)
        plt.clf()
        plt.subplot(121)
        plt.plot(*population[0].T)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.subplot(122)
        e = np.array(model(population[0]))
        e = e[:, 0]+1j*e[:, 1]
        P_get = np.angle(e)
        A_get = np.abs(e)
        plt.scatter(P_targets, P_get)
        plt.twinx()
        plt.scatter(P_targets, A_get, c='r')
        plt.ylim([0, max(A_get)*1.5])

        plt.pause(0.01)

print("DONE")
plt.show()


# Build and save to file -----------------------------------------
X_axis = X_axis.flatten()
Y_axis = Y_axis.flatten()

selected = []
for i in range(P_wants.shape[0]):
    for j in range(P_wants.shape[1]):
        selected.append(population[0][np.where(
            P_targets == P_wants[i, j])[0][0]])

selected = np.array(selected)
plt.plot(*selected.T)
plt.xlabel("W")
plt.ylabel("H")
plt.show()

with open("./template/multiplerect.ind", 'r') as f:
    text = f.read().format(
        SIZE=len(selected),
        width=size*period,
        height=size*period
    )

with open("./template/channel_segment.ind", "r") as f:
    segment_template = f.read()

segments = [[0, 0, 0, "width", "height", 0.1]]

for i in range(len(selected)):
    x = X_axis[i]
    y = Y_axis[i]
    z = 0.1
    w = f"{selected[i][0]}"
    h = f"{selected[i][1]}"
    segments.append([x, y, z, w, h, 0.8])


def make_segment(id, x, y, z, w, h, tall):
    return segment_template.format(
        id=id, x=x, y=y, z=z, w=w, h=h, tall=tall
    )


for i, segment in enumerate(segments):
    text += "\n\n"+make_segment(i+2, *segment)

with open("./multiple/multiplerect_2.ind", 'w') as f:
    f.write(text)
