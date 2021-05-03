import numpy as np
import matplotlib.pyplot as plt
import os

from most_write import mostmvwrite

W = []
H = []
TALL = []

# for tall in np.linspace(0.2, 8, 60):
#     n = 5

#     for w in np.linspace(0.2, 0.8, n):
#         for h in np.linspace(0.2, 0.8, n):
#             W.append(w)
#             H.append(h)
#             TALL.append(tall)

n = 40
for w in np.linspace(0.2, 0.8, n):
    for h in np.linspace(0.2, 0.8, n):
        W.append(w)
        H.append(h)

print(len(W))

if input("write file?") != 'y':
    exit()

mostmvwrite("./search_range.txt",
            ["W", "H"], W, H)
