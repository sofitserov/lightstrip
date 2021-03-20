import numpy as np

PIXELS = 300
p = np.tile(1, (3, PIXELS))

for i in range(PIXELS):
    print(p[0,i] + p[1,i] + p[2,i])
    pass
