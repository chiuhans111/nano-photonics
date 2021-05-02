from moviepy.editor import *
import numpy as np
from ifta_movie import pre_process, get_process_frame

print("program start")
clip = VideoFileClip("./movie/Movie studio Intros Compilation-pR3lNB_XI2k.mp4")

print(clip.fps)

def process_frame(gt, t):
    f = pre_process(gt, t)
    return np.uint8(f*255)[:, :, np.newaxis] * [[[1, 1, 1]]]

clip.fps = 10
fft_movie = clip.fl(get_process_frame())

fft_movie.write_videofile('./movie_doe.mp4')
