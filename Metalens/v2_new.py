import numpy as np
import matplotlib.pyplot as plt
from most_read import get_variable, get_result
from utils import angle_average
import tensorflow as tf


TRAIN_BASE_MODEL = False
BUILD_LENS = True


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
e = er+1j*ei
P = np.angle(e)
A = np.abs(e)
if False:
    plt.scatter(W, H, c=np.angle(er+1j*ei))
    plt.show()


# BASE MODEL ------------------------------------------------

def get_model():
    input_layer = tf.keras.Input([2])
    x = input_layer
    x = tf.concat([x, x[:, :1]*x[:, 1:]], 1)

    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(32, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
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
    pass
except:
    print("no model to load")

if TRAIN_BASE_MODEL:
    Ws = np.linspace(min(W), max(W), 50)
    Hs = np.linspace(min(H), max(H), 50)
    Ws, Hs = np.meshgrid(Ws, Hs)

    test_x = tf.stack([Ws.flatten(), Hs.flatten()], 1)

    for i in range(100):
        print(i)
        model.fit(train_x, train_y, batch_size=1024, epochs=50,
                  shuffle=True, verbose=0)
        model.save("./models/model.h5")
        model.save("./models/model_backup.h5")
        test_y = model(test_x)
        result = np.reshape(test_y, list(Ws.shape) + [2])
        r = result[::-1, :, 0]+1j*result[::-1, :, 1]
        plt.clf()
        plt.subplot(121)
        plt.imshow(np.angle(r),
                   extent=[min(W), max(W), min(H), max(H)])
        plt.subplot(122)
        plt.imshow(np.abs(r),
                   extent=[min(W), max(W), min(H), max(H)])
        plt.pause(0.01)
    plt.show()


# FIND ----------------------------------------------
if BUILD_LENS:

    size = 21
    period = 1
    fl = 20
    wl = 0.55

    # coordinates
    X_axis = np.linspace(-1, 1, size) / 2 * (size-1)*period
    Y_axis = np.linspace(-1, 1, size) / 2 * (size-1)*period
    X_axis, Y_axis = np.meshgrid(X_axis, Y_axis)

    P_wants = (fl-np.sqrt(fl**2+X_axis**2+Y_axis**2))/wl*np.pi*2 % (np.pi*2)

    P_targets = np.array(list(set(P_wants.flatten())))
    P_phasor = np.exp(-1j*P_targets)

    weight = []

    for p in P_targets:
        weight.append(np.sum(P_wants == p))

    print(weight)
    weight = np.array(weight)
    weight_sum = np.sum(weight)

    print(P_wants.shape, P_targets.shape)

    @tf.function
    def evaluate(selected):
        e = tf.cast(model(selected), tf.dtypes.complex64)
        phasor = (e[:, 0] + 1j*e[:, 1])
        phasor_signal = phasor * P_phasor
        background = tf.reduce_sum(phasor*weight)/weight_sum
        integrated = tf.reduce_sum(phasor_signal*weight)/weight_sum
        noise = tf.abs(background)
        signal = tf.abs(integrated)
        return signal/(noise+1)

    NP = 100

    population = []

    for i in range(NP):
        offset = i/NP*np.pi*2
        selected = []

        offset_phase = np.exp(offset*1j)

        for p in P_phasor:
            pp = e*p*offset_phase
            v = np.real(pp)-np.abs(np.imag(pp)) + \
                np.random.rand(pp.shape[0])*0.5
            best = np.argmax(v)
            selected.append([
                W[best],
                H[best]
            ])

        population.append(np.array(selected))
        # population.append(np.random.rand(P_targets.shape[0], 2)*0.6+0.2)

    last_score = []
    for iterations in range(1000):

        mutate = []
        for i in range(len(population)):
            j = np.random.randint(0, NP-1)
            k = np.random.randint(0, NP-2)
            l = np.random.randint(0, NP-3)

            if j >= i:
                j += 1

            if k >= min(i, j):
                k += 1
            if k >= max(i, j):
                k += 1

            ijk = np.sort([i, j, k])
            for n in ijk:
                if l >= n:
                    l += 1

            other1 = population[j]
            other2 = population[k]
            other3 = population[l]
            mask = np.random.rand(P_targets.shape[0], 2) < 0.5
            rnd = np.random.rand(P_targets.shape[0], 2)-0.5
            mutate.append(other1+(other2-other3+rnd*0.001)*(0.75)*mask)

        for i in range(len(mutate)):
            mutate[i] = np.clip(mutate[i], 0.2, 0.8)

        population2 = []
        score = []
        for i, selected in enumerate(mutate):
            s = evaluate(selected)

            if len(last_score) <= i or s > last_score[i]:
                population2.append(selected)
                score.append(s)
            else:
                population2.append(population[i])
                score.append(last_score[i])

        population = population2
        last_score = score

        if iterations % 10 == 0:
            print(iterations)
            best = np.argmax(score)

            plt.clf()
            plt.subplot(121)
            plt.plot(*population[best].T)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.subplot(122)
            e = np.array(model(population[best]))
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

    best = np.argmax(score)
    selected = []
    for i in range(P_wants.shape[0]):
        for j in range(P_wants.shape[1]):
            selected.append(population[best][np.where(
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
