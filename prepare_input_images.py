import cPickle
import sys
import tiff_to_npy
import os

in_list = sys.argv[1:]
out_list = []
for in_path in in_list:
    out_path = os.path.join("input", os.path.basename(in_path))
    tiff_to_npy.tiff_to_npy(in_path, out_path)
    out_list.append(out_path)
cPickle.dump(out_list, open("img.pkl","w"))
