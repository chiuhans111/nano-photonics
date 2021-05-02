from moviepy.editor import *
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageFilter
from numpy.lib.function_base import piecewise
from ifta import picture2doe
import numpy as np

print("program start")
clip = VideoFileClip("./cat-gif.mp4")

last_doe = 0
width = 256
cg_phase = None

print(clip.fps)


def process_frame(gf, t):
    global cg_phase, last_doe

    frame = gf(t)
    f = PIL.Image.fromarray(frame)
    # print(f.width, f.height)
    height = round(f.height/f.width*width/9*16)
    f = f.resize([width, height])
    # fc = f.filter(PIL.ImageFilter.CONTOUR())
    # fc = fc.filter(PIL.ImageFilter.MinFilter(size=3))
    f = np.mean(np.array(f)[:, :], 2)
    f -= np.min(f)

    # fc = np.array(fc)[:, :, 0]
    # f = 255-fc+f*0.1

    # print("frame", t)

    doe, cg, cg_phase = picture2doe(f, 100, cg_phase)
    # print(doe.shape)
    # print(np.min(doe))
    # print(np.max(doe))

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

    # print(d)

    doe = (doe - davg) % 256
    # print("err", np.mean(((doe-last_doe+128) % 256-128)**2))

    # im.save(f"./badapple_img/{i:04d}.jpg")
    last_doe = doe_sect
    return np.uint8(np.array(doe))[:, :, np.newaxis] * [[[1, 1, 1]]]


    # if i > 10:
    #     break
fft_movie = clip.fl(process_frame)
# fft_movie.write_videofile('./cat_doe.mp4')
fft_movie.write_gif('./cat_doe.gif')
