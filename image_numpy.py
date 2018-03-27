# Mahsa 
import cv2
import glob
import numpy as np
import os
import pdb
import ntpath #same as os.path
#Put all images in one numpy aarray

def images_to_numpyarray():
	X_data = []
	path=os.getcwd()+'/DRIVE/training/images/*.tif'
	files = glob.glob (path)


	for myFile in files:
	    print(myFile)
	    image = cv2.imread (myFile)
	    X_data.append (image)
	return X_data	
	    
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#pdb.set_trace()
#print('X_data shape:', np.array(X_data).shape)

#Make a list of image paths
def image_paths():
	paths = []
	gtpaths=[]
	path = os.getcwd() + '/DRIVE/training/images/*.tif'
	gtpath=os.getcwd()+'/DRIVE/training/1st_manual/*.gif'
	pathlist=glob.glob(path)
	gtpathlist=glob.glob(gtpath)
	for path_i in pathlist:
		image_name=ntpath.basename(path_i)
		dummy=os.path.join('input', image_name)
		paths.append(dummy)
	for path_j in gtpathlist:
		gt_name=ntpath.basename(path_j)
		dummygt=os.path.join('input', gt_name)
		gtpaths.append(dummygt)	
	return paths,gtpaths

def main():
	
	datapath,gtpath=image_paths()


if __name__=="__main__":
	
	main()


