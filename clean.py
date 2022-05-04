import numpy as np
import struct

with open('train-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

    import matplotlib.pyplot as plt

    plt.imshow(data[0], cmap='gray')
    plt.show()

    print(data[0])
