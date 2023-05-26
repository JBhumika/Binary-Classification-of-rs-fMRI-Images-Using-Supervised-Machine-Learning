import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import load_model


def search_the_fl(insid_fdr):
    reg = 'IC_.*thresh.png'
    extract_fls = []
    ct = 0
    folder = insid_fdr+'test_Data/'
    nme_of_the_all_fl = os.listdir(folder)
    for nme_of_fl in nme_of_the_all_fl:
        if re.match(reg, nme_of_fl):
            ct+=1
            extract_fls.append(folder+nme_of_fl)
    return extract_fls, ct


def crop(cped_img):
    cnt_ic = np.array(cped_img)
    for p in range(len(cnt_ic)):
        for q in range(len(cnt_ic[0])):
            if cnt_ic[p,q,0] > 100 and cnt_ic[p,q,2] < 200:
                cped_img = cped_img.crop((0,p,q,len(cnt_ic)))
                return cped_img


def imgld(extract_fls, mvmnt, ht, wd):
    ttl_ar = [0] * mvmnt
    for nme_of_fl in extract_fls:
        slt = int(nme_of_fl.split("_")[-2])-1
        cped_img = Image.open(nme_of_fl)
        cped_img = crop(cped_img)
        cped_img = cped_img.resize((wd, ht))
        ttl_ar[slt] = np.array(cped_img)
    return np.array(ttl_ar)


def axs_y_ld(insid_fdr):
    yfinal = []
    req_fle_nme = insid_fdr+'test_Labels.csv'
    rt = pd.read_csv(req_fle_nme)
    for u in rt['Label']:
        if u < 1:
            yfinal.append(0)
        else:
            yfinal.append(1)
    yfinal = np.array(yfinal)
    return yfinal


def final_mtrcs(mvmnt, yfinal, pred):
    c = confusion_matrix(yfinal, pred)
    accuracy = (c[0,0]+c[1,1]) / mvmnt
    precision = c[0,0]/(c[0,0]+c[1,0])
    sensitivity = (c[0,0])/(c[0,0]+c[0,1])
    specificity = (c[1,1])/(c[1,0]+c[1,1])
    print("The model Accuracy => %.2f" % (accuracy*100), "%")
    print("The model Precision => %.2f" % (precision*100), "%")
    print("The model Sensitivity => %.2f" % (sensitivity*100), "%")
    print("The model Specificity => %.2f" % (specificity*100), "%")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Sensitivity", "Specificity"],
        "Score": [np.round(accuracy*100, 2), np.round(precision*100, 2), np.round(sensitivity*100, 2), np.round(specificity*100, 2)]})
    metrics.to_csv('Metrics.csv', index=None)


def main():
    insid_fdr = 'testPatient/'
    ht = 384
    wd = 512

    extract_fls, mvmnt = search_the_fl(insid_fdr)
    print("The total amount of the data => ", mvmnt)

    xfinal = imgld(extract_fls, mvmnt, ht, wd)
    print("shape of x =>", xfinal.shape)

    yfinal = axs_y_ld(insid_fdr)
    print("shape of y =>", yfinal.shape)

    model = load_model('generatedmodel.h5')

    # predi-ctions
    pred = model.predict(xfinal)
    pred = (pred > 0.5).astype(np.float32)

    # Save the predictions
    idxes = [p for p in range(1,mvmnt+1)]
    res = pd.DataFrame({"IC_Number": idxes, "Label": pred.reshape(mvmnt)})
    res.to_csv('Results.csv', index=None)

    # Analyse the metrics
    final_mtrcs(mvmnt, yfinal, pred)


if __name__ == "__main__":
    main()

