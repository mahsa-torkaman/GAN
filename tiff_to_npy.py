#Mahsa: Change the all input images to .npy to dump it to .pkl file later 
import sys
from PIL import Image
import numpy
import pdb
from scipy import ndimage
import os
import glob
import image_numpy
import cPickle
import pickle
import ntpath

def tiff_to_npy(in_path, out_path):
    im = Image.open(in_path)
    #im.show()
    imarray = numpy.array(im)
    #pdb.set_trace()
    numpy.save(open("%s.npy" % (out_path.split(".")[0]),"w"), imarray)


def gif_to_npy(in_path,out_path):
    im_array = ndimage.imread(in_path)
    numpy.save(open("%s.npy" % (out_path.split(".")[0]),"w"), im_array)


def main():
	out_list=[]
	out_list_final=[]
	path1 = os.getcwd() + '/DRIVE/training/images/*.tif'
	in_path1=glob.glob(path1)
	path2=os.getcwd()+'/DRIVE/training/1st_manual/*.gif'
	in_path2=glob.glob(path2)
	in_path1_sorted=sorted(in_path1)
	in_path2_sorted=sorted(in_path2)
	i=0
	j=0
	x_paths,gt_paths=image_numpy.image_paths()

	for out_path in sorted(gt_paths):
		gif_to_npy(in_path2_sorted[j],out_path)
		j=j+1
		out_list.append(out_path+'.npy') #need that latel to make pikle

	for out_path in sorted(x_paths):
		tiff_to_npy(in_path1_sorted[i], out_path)
		i=i+1
		print(i)
		out_list.append(out_path+'.npy')
	out_list_sorted=sorted(out_list)
	for listi in out_list_sorted:
		vari=ntpath.basename(listi)
		bname=vari.split(".")[0]+"."+vari.split(".")[2]
		out_list_final.append("input/"+bname)
	pklobj1=[]#training data
	pklobj2=[]#segmentation data	

	for i in range(0,40,2):
		pklobj1.append(out_list_final[i])
		pklobj2.append(out_list_final[i+1])

	#make list of lists from out_list_final
	new_list=[]
	i=0
	while i<len(out_list_final):
		new_list.append(out_list_final[i:i+2])
		i+=2	
	#pdb.set_trace()
	#pickle.dump(out_list_final, open("img2.pkl","wb"))
	f=open("img3.pkl","w")
	cPickle.dump(new_list, f)# protocol=cPickle.HIGHEST_PROTOCOL)
	#f=open("img3.pkl","w")
	#pickle.dump(pklobj1,f)
	#pickle.dump(pklobj2,f)
	#f.close()

if __name__ == "__main__":
    main()   
