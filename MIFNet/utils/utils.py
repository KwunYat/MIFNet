import random
import numpy as np


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def batch(iterable, batch_size):
    """Yields lists by batch批量处理列表
    """
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        # print(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x // 255


