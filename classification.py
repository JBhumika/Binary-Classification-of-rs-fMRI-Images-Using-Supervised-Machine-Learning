import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D

def search_the_fl(insid_fdr):
    ic_img = 'IC_.*thresh.png'
    extract_fls = [[] for _ in range(5)]
    no_of_len = []
    for p in range(5):
        total_ct = 0
        folder = insid_fdr+'Patient_'+str(p+1)+'/'
        nme_of_the_all_fl = os.listdir(folder)
        for nme_of_fl in nme_of_the_all_fl:
            if re.match(ic_img, nme_of_fl):
                total_ct+=1
                extract_fls[p].append(folder+nme_of_fl)
        no_of_len.append(total_ct)
    return extract_fls, no_of_len

def crop(cped_img):
    cnt_ic = np.array(cped_img)
    for p in range(len(cnt_ic)):
        for q in range(len(cnt_ic[0])):
            if cnt_ic[p,q,0] > 100 and cnt_ic[p,q,2] < 200:
                cped_img = cped_img.crop((0,p,q,len(cnt_ic)))
                return cped_img

def imgld(extract_fls, mvmnt, ht, wd):
    ttl_ar = [0] * mvmnt
    total_ct = 0
    for p,nams_of_fl in enumerate(extract_fls):
        new_ct = 0
        for nme_of_fl in nams_of_fl:
            slt = int(nme_of_fl.split("_")[-2])-1
            cped_img = Image.open(nme_of_fl)
            cped_img = crop(cped_img)
            cped_img = cped_img.resize((wd, ht))
            ttl_ar[slt+total_ct] = np.array(cped_img)
            new_ct +=1
        total_ct += new_ct
    return np.array(ttl_ar)


def axs_y_ld(insid_fdr):
    yfinal = []
    for p in range(5):
        req_fle_nme = insid_fdr+'Patient_'+str(p+1)+'_Labels.csv'
        rt = pd.read_csv(req_fle_nme)
        for u in rt['Label']:
            if u < 1:
                yfinal.append(0)
            else:
                yfinal.append(1)
    yfinal = np.array([[u] for u in yfinal])
    return yfinal


def mdl_bld(ht, wd, cnl):
    final_m = Sequential()
    final_m.add(Conv2D(16, (3,3), input_shape=(384, 512, 3)))
    final_m.add(MaxPooling2D((2,2)))
    final_m.add(Conv2D(32, (3,3), activation='relu'))
    final_m.add(MaxPooling2D((2,2)))
    final_m.add(Conv2D(64, (3,3), activation='relu'))
    final_m.add(MaxPooling2D((2,2)))
    final_m.add(Conv2D(64, (3,3), activation='relu'))
    final_m.add(MaxPooling2D((2,2)))
    final_m.add(Conv2D(128, (3,3), activation='relu'))
    final_m.add(MaxPooling2D((2,2)))
    final_m.add(Flatten())
    final_m.add(Dense(32))
    final_m.add(Dense(1, activation='sigmoid'))
    return final_m


def main():
    wd = 512
    insid_fdr = 'PatientData/'
    ht = 384
    cnl = 3

    extract_fls, no_of_len = search_the_fl(insid_fdr)
    mvmnt = sum(no_of_len)
    print("The total amount of the data => ", mvmnt)
    print("The total number of data per patient => ", no_of_len)
    
    xfinal = imgld(extract_fls, mvmnt, ht, wd)
    print("x shape => ", xfinal.shape)

    yfinal = axs_y_ld(insid_fdr)
    print("y shape => ", yfinal.shape)

    # Split tha data train model and num of x and y
    trn_mdl_x, numx, trn_mdl_y, numy = train_test_split(xfinal, yfinal, test_size=0.05)

    # Print all the shapes
    print()
    print("train model x shp => ", trn_mdl_x.shape)
    print("train model y shp => ", trn_mdl_y.shape)
    print("val of x shp => ", numx.shape)
    print("val of y shp => ", numy.shape)

    final_m = mdl_bld(ht, wd, cnl)

    final_m.compile(loss='binary_crossentropy', optimizer='Adam', metrics='accuracy')
    print(final_m.summary())

    epochs = 20
    # Train-ing
    bs = 5
    final_m.fit(trn_mdl_x, trn_mdl_y, validation_data=(numx, numy), batch_size=bs, epochs=epochs)

    # Save the model as generatedmodel.h5
    final_m.save('generatedmodel.h5')

if __name__ == "__main__":
    main()