import os
import numpy as np
from PIL import Image
from .utils import normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=1):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, j) for id in ids for j in range(n))


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_nir = Image.open(dir + 'train_nir/' + 'nir' + id[2:] + suffix)
        img_nir = np.array(img_nir)
        img_nir = img_nir[np.newaxis, ...]
        yield img_nir


def to_cropped_mask(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_mask = Image.open(dir + id + suffix)
        mask = np.array(img_mask)
        mask = mask/255
        yield mask
        
def to_cropped_edge(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_edge = Image.open(dir + 'edge' + id[2:] + suffix)
        edge = np.array(img_edge)
        edge = edge/255
        yield edge


def get_imgs_and_masks_and_edges(ids, dir_img, dir_mask, dir_edge):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.TIF')
    masks = to_cropped_mask(ids, dir_mask, '.TIF')
    edges = to_cropped_edge(ids, dir_edge, '.TIF')

    return zip(imgs, masks, edges)


def get_full_img_and_mask_and_edges(id, dir_img, dir_mask, dir_edge):
    im = Image.open(dir_img + id + '.png')
    mask = Image.open(dir_mask + id + '.png')
    edge = Image.open(dir_edge + id + '.png')
    return np.array(im), np.array(mask), np.array(edge)


def get_test_img(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_nir = Image.open(dir + 'test_nir/' + 'nir' + id[3:] + suffix)
        img_nir = np.array(img_nir)
        img_nir = img_nir[np.newaxis, ...]
        yield [img_nir, id]
