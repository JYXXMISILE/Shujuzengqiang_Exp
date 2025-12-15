# check_npy_shapes.py
import os
import numpy as np

def check_shapes(directory):
    shapes = []
    for fn in os.listdir(directory):
        if fn.endswith('.npy'):
            path = os.path.join(directory, fn)
            data = np.load(path)
            shapes.append(data.shape)
    return shapes

train_dir = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\train_audio"
val_dir = r"E:\cat的方法\T192_mel163\所有音频集合_dataset\val_audio"

train_shapes = check_shapes(train_dir)
val_shapes = check_shapes(val_dir)

print("Train shapes:", set(train_shapes))
print("Val shapes:", set(val_shapes))