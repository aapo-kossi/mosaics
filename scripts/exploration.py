import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageOps import exif_transpose
from glob import glob
import matplotlib.pyplot as plt
import pathlib
import multiprocessing
import os
import argparse
import random
from scipy import spatial

BIN_WIDTH = 3
PIC_PATH = "C:\\Users\Aapo\Downloads\colorful-umbrella-1176220.jpg"

def shard(data, n):
    datalen = len(data)
    shardlen = datalen // n
    assert shardlen > 0
    prev = 0
    shards = []
    while True:
        fin = min([datalen+1,prev+shardlen])
        shards.append(data[prev:fin])
        prev += shardlen
        if fin == datalen+1: break
    return shards

def save_shard(names):
    means = np.zeros((len(names),3))
    r_lookup = get_empty_lookup()
    for n, filename in enumerate(names):
        means[n] = np.mean(Image.open(filename).getdata(), 0)
        bin = get_bins(means[n])
        r_lookup[tuple(bin)].append(filename)
    return means, r_lookup

def get_bins(rgb):
    return (rgb // BIN_WIDTH)

def get_empty_lookup():
    base = np.arange(256 // BIN_WIDTH)
    length = base.shape[0]
    base = np.tile(base, (length,1))
    base = np.tile(base, (length,1,1))
    base = np.stack((base,np.transpose(base,(1,2,0)),np.transpose(base,(2,0,1))),3).reshape(-1,3)
    empty = {tuple(k): [] for k in base}
    return empty

# def get_img_with_close_clr(init_bin, rlookup):
#     done = False
#     to_check = [init_bin]
#     while true:
#         candidates = []
#         for bin_cand in to_check:
#             candidates.extend(rlookup[bin_cand])
#         if candidates: break
#         for prev_bin in to_check:
#             if prev_bin[0] >= init_bin[0]: to_check.append((prev_bin[0]+1,)+prev_bin[1:])
#             if prev_bin[0] <= init_bin[0]: to_check.append((prev_bin[0]-1,)+prev_bin[1:])
#             if prev_bin[1] >= init_bin[1]: to_check.append((prev_bin[0],)+(prev_bin[1]+1,)+prev_bin[2:])
#             if prev_bin[1] <= init_bin[1]: to_check.append((prev_bin[0],)+(prev_bin[1]-1,)+prev_bin[2:])
#             if prev_bin[2] >= init_bin[2]: to_check.append((prev_bin[:2],)+(prev_bin[2]+1,)+prev_bin[3:])
#             if prev_bin[2] <= init_bin[2]: to_check.append((prev_bin[:2],)+(prev_bin[2]-1,)+prev_bin[3:])
#             if prev_bin[3] >= init_bin[3]: to_check.append(prev_bin[:3]+(prev_bin[3]-1,))
#             if prev_bin[3] <= init_bin[3]: to_check.append(prev_bin[:3]+(prev_bin[3]-1,))
#         to_check.remove(prev_bin)
#         to_check = list(dict.fromkeys(to_check))
#     return random.sample(candidates,1)[0]

def crop(im,h,w):
    imgwidth, imgheight = im.size
    for i in range(imgheight//h):
        for j in range(imgwidth//w):
            box = (j*w, i*h, (j+1)*w, (i+1)*h)
            yield im.crop(box)

def main():
    smsz = 15
    tgt_ratio = 2
    n_cpu = multiprocessing.cpu_count()
    datapath = pathlib.Path(os.getcwd()).parent / "data/images/*.jpg"
    img_names = glob(str(datapath))
    sharded_names = shard(img_names, n_cpu)
    p = multiprocessing.Pool(processes=n_cpu)
    results = [p.apply_async(save_shard, args = (x,)) for x in sharded_names]
    results = [item.get() for item in results]
    sharded_means, sharded_lookups = zip(*results)
    pixel_means = np.concatenate(sharded_means,0)
    tree = spatial.KDTree(pixel_means)
#    reverse_lookup = sharded_lookups[0]
#    for orig_dict in sharded_lookups[1:]:
#        for key, value in orig_dict.items():
#            reverse_lookup[key].extend(value)

    big_path = PIC_PATH
    big_img = exif_transpose(Image.open(big_path))
    imw, imh = big_img.size
    n_col = imw // smsz
    n_row = imh // smsz
    crop_generator = crop(big_img, smsz, smsz)
    patch_imgs = np.zeros(shape=(n_row,n_col),dtype=np.dtype('U128'))
    for i in range(n_row):
        for j in range(n_col):
            patch = next(crop_generator)
            picked = random.randint(0,19)
            idx = tree.query(np.mean(patch.getdata(),0),k=20)[1][picked]
            patch_imgs[i,j] = img_names[idx]
    del big_img
    del big_path
    del pixel_means
    del results
    del sharded_lookups
    del sharded_means
    del sharded_names

    canvas = Image.new('RGB', (n_col*smsz*tgt_ratio,n_row*smsz*tgt_ratio))
    for i in range(n_row):
        for j in range(n_col):
            im = Image.open(patch_imgs[i,j])
            x, y = j*smsz*tgt_ratio, i*smsz*tgt_ratio
            im = im.resize((smsz*tgt_ratio,smsz*tgt_ratio), resample=Image.Resampling.LANCZOS)
            canvas.paste(im, (x,y))

    canvas.save(str(pathlib.Path(os.getcwd()).parent / "sateenvarjo_worsened.jpg"))



if __name__ == "__main__":
    main()