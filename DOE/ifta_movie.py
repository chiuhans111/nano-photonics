from moviepy.editor import *
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageFilter
from numpy.ma.core import maximum_fill_value
from ifta import picture2doe
import numpy as np

last_doe = 0
width = 256
cg_phase = None

height = None
def get_height(f):
    return round(f.height/f.width*width/9*16)


def pre_process(gf, t):
    global height
    frame = gf(t)
    f = PIL.Image.fromarray(frame)

    height = get_height(f)
    f = f.resize([width, height])
    f = np.array(f)/255

    minimum = np.min(np.min(f, 0, keepdims=True), 1, keepdims=True)
    maximum = np.max(np.max(f, 0, keepdims=True), 1, keepdims=True)
    value_range = maximum - minimum

    value_range = value_range+(0.1-value_range)*(value_range < 0.1)

    f -= minimum
    f = f/value_range

    f = np.mean(f, 2)
    f -= np.min(f)

    maximum = np.max(f)
    if maximum < 0.1:
        maximum = 0.1
    f /= maximum

    f = f**2

    # plt.imshow(f)
    # plt.pause(0.01)
    return f


def get_process_frame():
    def process_frame(gf, t):
        global cg_phase, last_doe

        f = pre_process(gf, t)

        doe, cg, cg_phase = picture2doe(f, 100, cg_phase)

        doe_sect = doe[:width, :height]
        d = np.array(doe_sect-last_doe).flatten()/256*2*np.pi
        x = np.cos(d)
        y = np.sin(d)
        xavg = np.mean(x)
        yavg = np.mean(y)
        dist = (x-xavg)**2+(y-yavg)**2
        mask = dist < 0.1

        if np.any(mask):
            xavg = np.mean(x[mask])
            yavg = np.mean(y[mask])
        davg = np.arctan2(yavg, xavg)/2/np.pi*256

        doe = (doe - davg) % 256
        last_doe = doe_sect
        return np.uint8(np.array(doe))[:, :, np.newaxis] * [[[1, 1, 1]]]
    return process_frame
