# -*- coding: utf-8 -*-


import os
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import GridSearchCV
# Divide up into cars and notcars


notcars = glob.glob('D:/neg/GIT/*.bmp')
cars = glob.glob('D:/pos/*.bmp')






colorspace = 'YUV' # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # 0, 1, 2, or "ALL"

t = time.time()
car_features = utils.extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = utils.extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

#利用numpy库生成一个特征向量
X = np.vstack((car_features, notcar_features))
X = X.astype(np.float64)                       


#利用numpy生成特征向量的标签向量
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


#随机设置测试数据集与正负数据及比例
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)


print('Feature vector length:', len(X_train[0]))
#使用SVC 创建SVC对象
svc = LinearSVC()
# 进行训练
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train classfier...')
# 预估一下测试准确率
print('Test Accuracy of classfier = ', round(svc.score(X_test, y_test), 4))
# 对测试数据集预估训练结果
t=time.time()
n_predict = 10
print('My classfier predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with classfier')

#配置训练结果参数
train_dist={}
train_dist['clf']=svc
train_dist['scaler']=None
train_dist['orient']=orient
train_dist['pix_per_cell'] = pix_per_cell
train_dist['cell_per_block'] = cell_per_block
train_dist['hog_channel'] = hog_channel
train_dist['spatial_size'] = None
train_dist['hist_bins'] = None
#保存训练结果
output = open('train_dist.p', 'wb')
pickle.dump(train_dist,output)

