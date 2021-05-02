from moviepy.editor import *
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageFilter
import numpy as np


print("program start")
clip = VideoFileClip("./movie_doe.mp4")

print(clip.fps)


for i, frame in enumerate(clip.iter_frames(fps=15)):

    f = PIL.Image.fromarray(frame)
    f = np.exp(np.array(f)[:, :, 0]*2j*np.pi/256)
    result = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f))))**2
    plt.clf()
    plt.title(i)
    plt.imshow(result[::-1, ::-1])
    plt.pause(0.02)

    # if i > 10:
    #     break
