from PIL import Image
import numpy
import cPickle
import sys
import os
from scipy import misc
import scipy.io

masks_out_path = 'syn_data/mask_trn20+tst15+flip'
masks_path = '/Users/Mahsa/Downloads/code/code/DRIVE/training/mask/'

size = 512, 512


training_images_out_path = 'trn20+tst20.mat'
training_images_path = '/Users/Mahsa/Downloads/code/code/DRIVE/training/images/'

in_list = [x for x in os.listdir(masks_path)]
all_masks_array = []
for in_path in sorted(in_list):
    im = Image.open(os.path.join(masks_path,in_path))
    im = im.resize(size, Image.ANTIALIAS)
    #im.show()
    print in_path
    imarray = numpy.array(im)
    all_masks_array.append(imarray)
numpy.save(open("%s.npy" % (masks_out_path),"w"), all_masks_array)

training_out_dict = {'allImgTrain': [] }


in_list = [x for x in os.listdir(training_images_path)]
all_images_array = []
for in_path in sorted(in_list):
    im = Image.open(os.path.join(training_images_path,in_path))
    im = im.resize(size, Image.ANTIALIAS)
    #im.show()
    print in_path
    imarray = numpy.array(im)
    all_images_array.append(imarray)
training_out_dict['imgAllTrain'] = all_images_array
scipy.io.savemat(training_images_out_path, training_out_dict)
#raster = misc.imread('image.tif')
