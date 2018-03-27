#Make the masks logical 0,1
import pdb
import numpy as np 
masks=np.load('mask_trn20+tst15+flip.npy')
mask_trn20=[]
for i in range(20):
	#pdb.set_trace()
	img=masks[i]
	maskrescaled=img/np.max(img)
	mask_trn20.append(maskrescaled)
np.save('mask_trn20_scaled',mask_trn20)