from __future__ import print_function
import cv2
import time
import glob
import numpy as np
import os
import math
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, current_process
import multiprocessing

MIN_MATCH_COUNT = 10
height, width = [], []
t = 0

RGB = '/path to the directory storing RGB images/'
CIR = '/path to the directory storing NIR images/'

if not os.path.isdir(RGB):
    os.mkdir(RGB)

if not os.path.isdir(CIR):
    os.mkdir(CIR)


def Homography_Matrix(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect SIFT features and compute descriptors.
    sift = cv2.SIFT_create(nfeatures=1000000000)
    
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        H = None
    return H

def alignment(r, g):
    global procs

    for j, k in zip(r, g):  
        cir = cv2.imread(j); rgb = cv2.imread(k)   
        red, green, blue, nir, rededge = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], cir[:,:,0], cir[:,:,1]
        red = cv2.cvtColor(red,cv2.COLOR_GRAY2RGB)
        green =cv2.cvtColor(green,cv2.COLOR_GRAY2RGB)
        blue = cv2.cvtColor(blue,cv2.COLOR_GRAY2RGB)
        nir = cv2.cvtColor(nir, cv2.COLOR_GRAY2RGB)
        rededge = cv2.cvtColor(rededge, cv2.COLOR_GRAY2RGB)

        print("Aligning Rededge images ...")
        
        rededge_Reg = alignImages(nir, rededge)
        nir_Reg = alignImages(rededge_Reg, nir)
        G_Reg = alignImages(nir, green)
        R_Reg = alignImages(G_Reg, red)
        B_Reg = alignImages(R_Reg, blue)
        nir_Reg = alignImages(B_Reg, nir_Reg)

        R_Reg = cv2.cvtColor(R_Reg, cv2.COLOR_BGR2GRAY)
        G_Reg = cv2.cvtColor(G_Reg, cv2.COLOR_BGR2GRAY)
        B_Reg = cv2.cvtColor(B_Reg, cv2.COLOR_BGR2GRAY)
        nir_Reg = cv2.cvtColor(nir_Reg, cv2.COLOR_BGR2GRAY)
        rededge_Reg = cv2.cvtColor(rededge_Reg, cv2.COLOR_BGR2GRAY)

        h, w = R_Reg.shape; height.append(h), width.append(w)
        h, w = G_Reg.shape; height.append(h), width.append(w)
        h, w = B_Reg.shape; height.append(h), width.append(w)
        h, w = nir_Reg.shape; height.append(h), width.append(w)
        h, w = rededge_Reg.shape; height.append(h), width.append(w)
        
        # keep the size of all bands equal after stacking channels
        Smallest_h = min(height); Smallest_w = min(width)
        G_Reg = G_Reg[G_Reg.shape[0]-Smallest_h:Smallest_h+(G_Reg.shape[0]-Smallest_h), G_Reg.shape[1]-Smallest_w:Smallest_w+(G_Reg.shape[1]-Smallest_w)]
        B_Reg = B_Reg[B_Reg.shape[0]-Smallest_h:Smallest_h+(B_Reg.shape[0]-Smallest_h), B_Reg.shape[1]-Smallest_w:Smallest_w+(B_Reg.shape[1]-Smallest_w)]
        R_Reg = R_Reg[R_Reg.shape[0]-Smallest_h:Smallest_h+(R_Reg.shape[0]-Smallest_h), R_Reg.shape[1]-Smallest_w:Smallest_w+(R_Reg.shape[1]-Smallest_w)]
        nir_Reg = nir_Reg[nir_Reg.shape[0] - Smallest_h:Smallest_h + (nir_Reg.shape[0] - Smallest_h), nir_Reg.shape[1] - Smallest_w:Smallest_w + (nir_Reg.shape[1] - Smallest_w)]
        rededge_Reg = rededge_Reg[rededge_Reg.shape[0] - Smallest_h:Smallest_h + (rededge_Reg.shape[0] - Smallest_h), rededge_Reg.shape[1] - Smallest_w:Smallest_w + (rededge_Reg.shape[1] - Smallest_w)]


        rgb = np.dstack([R_Reg, G_Reg, B_Reg])
        rgb = crop_edges(rgb)
        
        cir = np.dstack([nir_Reg, nir_Reg, nir_Reg])
        cir = crop_edges(cir)
               
        fineSize_h, fineSize_w = 800, 1100
        w_offset = int(np.random.uniform(0, max(0, w - fineSize_w - 1)))
        h_offset = int(np.random.uniform(0, max(0, h - fineSize_h - 1)))

        rgb = rgb[h_offset:h_offset + fineSize_h, w_offset:w_offset + fineSize_w]
        cir = cir[h_offset:h_offset + fineSize_h, w_offset:w_offset + fineSize_w]
                
        print ('Aliging MultiSpectral image ...')
        # second image registration after stacking channels to keep images pixel wise aligned
    ############################################################################ 
        h1, w1 = rgb.shape[:2]
        im_ = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_CUBIC)
        cirReg = alignImages(im_, cir)
        if cirReg is None:
            continue
        cirReg = crop_edges(cirReg)
   #############################################################################    
        redReg = alignImages(cirReg, rgb)
        if redReg is None:
            continue
        redReg = crop_edges(redReg)
    ############################################################################        
        process_name = current_process().name      
        
        print('Storing Aligned Images ...')
        cv2.imwrite(RGB + '{0}.png'.format(t), redReg)
        cv2.imwrite(CIR + '{0}.png'.format(t), cirReg)
        t += 1
        print(f"Process Name: {process_name}")

def alignImages(im1, im2):
    M = Homography_Matrix(im1, im2)
    if M is None:
        return
    h, w = im1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
    rededge = cv2.warpPerspective(im2, perspectiveM, (w, h))  

    return rededge

def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]

if __name__ == '__main__':

    imagenames_list = []
    chunk_size = 400  #number of images
    procs = 10 #number of cores

    cir_images = sorted(glob.glob('path to the directory of color calibrated NIR images'))
    rgb_images = sorted(glob.glob('path to the directory of color calibrated RGB images'))
    
    r, g, b, n, re = [],[],[],[],[]
    for j, k in (zip(cir_images, rgb_images)):   
        r.append(j); g.append(k)
    
    #Multiprocessing image alignment
    processes = []
    for i in range(0, procs):
        print (i)
        process = Process(target=alignment, args=(r, g,))
        processes.append(process)
        process.start()             
    for proc in processes:
        proc.join()
    print('Multiprocessing Completed!')
    
