import os
import glob
import csv
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

seed = 123
np.random.seed(seed)

def processing(addrs):

	data = []
	# loop over all addresses
	for i in range(len(addrs)):
		addr = addrs[i]
		img = cv2.imread(addr)
		img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img/255
		assert(img.shape == (224,224,3))
		data.append(img)

	data = np.asarray(data)
	assert(data.shape == (len(addrs),224,224,3))
	return data

def create_train_test_splits(path):

    image_files = []
    for folder1 in sorted(os.listdir(path)):
        dir1 = path + '/' + folder1
        for folder2 in sorted(os.listdir(dir1)):
            dir2 = dir1 + '/' + folder2
            for folder3 in sorted(os.listdir(dir2)):
                dir3 = dir2 + '/' + folder3
                if folder3 == 'Occupied':
                    for file in sorted(glob.glob(os.path.join(dir3, '*.jpg'))):
                        image_files += [[file,1,0]]
                        
                else:
                    for file in sorted(glob.glob(os.path.join(dir3, '*.jpg'))):
                        image_files += [[file,0,1]]
    
    df = pd.DataFrame(image_files,columns=['path_to_image','occupied','free'])
    df = df.sample(frac=1,random_state=seed).reset_index(drop=True)

    X = df['path_to_image'].get_values()
    Y = df.drop(columns=['path_to_image']).get_values()

    X_train,X_test,Y_train,Y_test = train_test_split(X , Y,test_size=0.5, stratify= Y,random_state=seed)

    train_data = pd.DataFrame({'path_to_image': X_train,'occupied': Y_train[:,0], 'free':Y_train[:,1]})
    test_data = pd.DataFrame({'path_to_image': X_test, 'occupied': Y_test[:,0], 'free':Y_test[:,1]})

    train_data.to_csv('train_set.csv',index=False)
    test_data.to_csv('test_set.csv',index=False)


def main():

	working_directory = os.getcwd()
	path_to_dataset = working_directory + '/data/PKLot/PKLotSegmented/PUC'
	create_train_test_splits(path_to_dataset)

if __name__ == '__main__':
	main()
