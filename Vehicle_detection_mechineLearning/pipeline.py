import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import utils
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


dist_pickle = pickle.load(open("train_dist.p", "rb"))
svc = dist_pickle["clf"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]




# 定义一个函数，该函数可以使用HOG提取特征并进行预测
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    windows = []

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = utils.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = utils.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = utils.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell



            test_prediction = svc.predict(hog_features.reshape(1,-1))

            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                windows.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return windows
def search_car(img):
    draw_img = np.copy(img)

    windows = []

    colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    ystart = 400
    ystop = 464
    scale = 1.0
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 416
    ystop = 480
    scale = 1.0
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 496
    scale = 1.5
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 528
    scale = 1.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 528
    scale = 2.0
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 560
    scale = 2.0
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 596
    scale = 3.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 464
    ystop = 660
    scale = 3.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    
#    window_list = utils.slide_window(img)

    
    heat_map = np.zeros(img.shape[:2])
    heat_map = utils.add_heat(heat_map,windows)
    heat_map_thresholded = utils.apply_threshold(heat_map,1)
    labels = label(heat_map_thresholded)
    draw_img = utils.draw_labeled_bboxes(draw_img,labels)

    return draw_img

ystart = 400
ystop = 656
scale = 1.5
'''
test_imgs=[]
out_imgs = []
img_paths = glob.glob('test_images/*.jpg')
plt.figure(figsize=(20,68))
for path in img_paths:
   img = mpimg.imread(path)
   out_img = search_car(img)
   test_imgs.append(img)
   out_imgs.append(out_img)

plt.figure(figsize=(20,68))
for i in range(len(test_imgs)):

   plt.subplot(2*len(test_imgs),2,2*i+1)
   plt.imshow(test_imgs[i])

   plt.subplot(2*len(test_imgs),2,2*i+2)
   plt.imshow(out_imgs[i])
'''

project_outpath = 'vedio_out/project_video_out.mp4'
project_video_clip = VideoFileClip("project_video.mp4")
project_video_out_clip = project_video_clip.fl_image(search_car)
project_video_out_clip.write_videofile(project_outpath, audio=False)
