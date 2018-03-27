# Mahsa
import cPickle
import sys
import tiff_to_npy
import os
import image_numpy
import pdb
import glob
#in_list = sys.argv[1:]

out_list = []
x_paths=image_numpy.image_paths()
#pdb.set_trace()
path = os.getcwd() + '/DRIVE/training/images/*.tif'
in_path=glob.glob(path)
i=0
for out_path in x_paths:
    #pdb.set_trace()
    #out_path = os.path.join("input", os.path.basename(in_path))
    tiff_to_npy.tiff_to_npy(in_path[i], out_path)
    #
    i=i+1
    out_list.append(out_path+'.npy')
cPickle.dump(out_list, open("img1.pkl","w"))
