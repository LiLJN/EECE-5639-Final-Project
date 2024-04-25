import numpy as np
import cv2
import os
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def MSE(path):
    files=[]
    for filename in os.listdir(path):
        files.append(filename)
    MSE = 0
    for i in range(1,len(files)):
        img_path_ref = path + '/' + files[i-1]
        img_path_new = path + '/' + files[i]
        #print(img_path_ref)
        #print(img_path_new)
        img_ref = cv2.imread(img_path_ref)
        img_new = cv2.imread(img_path_new)
        ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        height, width = ref_gray.shape
        ref_data = []
        new_data = []

        for i in range(0, height):
            for j in range(0, width):
                ref_data.append(ref_gray[i][j])
                new_data.append(new_gray[i][j])

        mse = mean_squared_error(ref_data, new_data)
        '''
        for i in range(0, height,400):
            for j in range(0, width, 300):
                ref_data = ref_gray[i][j]
                new_data = new_gray[i][j]
                mse += ((ref_data-new_data)**2) / (height*width)
        mse = mse
        '''
        MSE += mse

    return MSE/len(files)


def MAE(path):
    files=[]
    for filename in os.listdir(path):
        files.append(filename)
    MAE = 0
    for i in range(1,len(files)):
        img_path_ref = path + '/' + files[i-1]
        img_path_new = path + '/' + files[i]
        #print(img_path_ref)
        #print(img_path_new)
        img_ref = cv2.imread(img_path_ref)
        img_new = cv2.imread(img_path_new)
        ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        height, width = ref_gray.shape
        ref_data = []
        new_data = []

        for i in range(0, height):
            for j in range(0, width):
                ref_data.append(ref_gray[i][j])
                new_data.append(new_gray[i][j])

        mae = mean_absolute_error(ref_data, new_data)
        MAE += mae
    return MAE/len(files)

def SSIM(path):
    files=[]
    for filename in os.listdir(path):
        files.append(filename)
    SSIM = 0
    for i in range(1,len(files)):
        img_path_ref = path + '/' + files[i-1]
        img_path_new = path + '/' + files[i]
        #print(img_path_ref)
        #print(img_path_new)
        img_ref = cv2.imread(img_path_ref)
        img_new = cv2.imread(img_path_new)
        ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        height, width = ref_gray.shape
        ssim_score, dif = ssim(ref_gray, new_gray, full=True)
        SSIM += ssim_score

    return SSIM/len(files)


SSIM_new = SSIM('H:/study/CV/Project-2/MSE_NEW')
SSIM_old = SSIM('H:/study/CV/Project-2/MSE_OLD')
print('The SSIM of original video : ' + str(SSIM_old))
print('The SSIM of revised video  : ' + str(SSIM_new))
print('Bigger SSIM -> Better performance')

MAE_new = MAE('H:/study/CV/Project-2/MSE_NEW')
MAE_old = MAE('H:/study/CV/Project-2/MSE_OLD')
print('The MAE of original video : ' + str(MAE_old))
print('The MAE of revised video  : ' + str(MAE_new))
print('Small MAE -> Better performance')

MSE_new = MSE('H:/study/CV/Project-2/MSE_NEW')
MSE_old = MSE('H:/study/CV/Project-2/MSE_OLD')
print('The MSE of original video : ' + str(MSE_old))
print('The MSE of revised video  : ' + str(MSE_new))
print('Small MSE -> Better performance')