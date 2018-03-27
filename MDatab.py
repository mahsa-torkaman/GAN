#Mahsa.
import numpy as np
import pdb
import cPickle
import pickle

def load_func(path):
        arr = np.load(open(path))
        return arr

def load_file(paths):
        
        train = []
        valid = []
        datablock=[]
        if isinstance(paths, (list, tuple)):
            for path in paths:
            	#pdb.set_trace()
                X = load_func(path[0])
                y=load_func(path[1])
                #pdb.set_trace()
                #num_train = len(X) * self.train_valid_ratio[0] * 1.0 / sum(self.train_valid_ratio)
                #num_train = int(num_train)
                #train.append(X[:num_train])
                #valid.append(X[num_train:])
                train.append(X)
                valid.append(y)
                #data=[train,valid]
                #datablock.append(data)
            #pdb.set_trace()        
        return train,valid


def main():
        temp=[]
        db=[]
        readpath = open('img2.pkl', 'rb')
        #pdb.set_trace()
        datapaths = cPickle.load(readpath)
        i=0
        new_list=[]#break the datapath list to a list of lists
        while i<len(datapaths):
          new_list.append(datapaths[i:i+2])
          i+=2
        #pbd.set_trace()
        t,v=load_file(new_list)
        return [t,v]

if __name__=="__main__":
	main()
