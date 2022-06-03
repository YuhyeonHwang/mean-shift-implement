import cv2
import numpy as np
import copy
import time

epsilon = 0.1

def imgResize(img,rate=0.5):
    return cv2.resize(img, dsize=(0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)

def eucliDist(pt1,pt2):
    return np.sqrt(np.sum(np.square(pt1-pt2)))

def gaussianKernel(x,bandwidth):
    xValue = np.sqrt(np.sum(np.square(x)))/bandwidth
    if xValue<=1:
        return np.exp((-np.square(xValue)))
    else:
        return 0

def gaussianKernelPos(x,bandwidth):
    '''
    '''
    xValue = np.sqrt(np.sum(np.square(x)))/bandwidth
    if xValue<=1:
        xRGB = np.sqrt(np.sum(np.square(x[:3])))/bandwidth
        xPos = np.sqrt(np.sum(np.square(x[3:])))/bandwidth*1.2
        return np.exp((-np.square(xRGB)))*np.exp((-np.square(xPos)))
    else:
        return 0

def meanShift(img,bandwidth=5):
    '''
    '''
    ### RGB difference value is not linear from human's cognitive perspective -> Luv domain
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
        img = np.array([img[:,:,0],img[:,:,1],img[:,:,2]])
    elif len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2Luv)
        img = img.reshape((1,)+img.shape)
    height,width = np.array((img[0,:,:],img[0,:,:])).shape
    pixelLocation = np.zeros(shape=np.array((img[0,:,:],img[0,:,:])).shape,dtype=np.uint16)
    for y in range(height):
        for x in range(width):
            pixelLocation[:,y,x] = [y,x]
    pixelData = np.concatenate((img,pixelLocation))
    modeCurrent = pixelData
    modeUpdate = 0
    mode = np.zeros(shape=modeCurrent.shape,dtype=np.float32)
    condition = 1
    starttimeTotal = time.time()
    print("height : ", height)
    for y in range(height):
        print(y)
        starttime = time.time()
        for x in range(width):
            condition = 1
            while condition==True:
                xkSum = 0
                kSum = 0
                for modey in range(height):
                    for modex in range(width):
                        kValue = gaussianKernelPos((pixelData[:,modey,modex]-modeCurrent[:,y,x]),bandwidth)
                        xkSum += pixelData[:,modey,modex]*kValue
                        kSum += kValue
                modeUpdate = xkSum/kSum
                criterion = eucliDist(modeUpdate,modeCurrent[:,y,x])
                # print("L1 norm (mode) : ",criterion)
                if epsilon<0.01:
                    condition = 0
                modeCurrent[:,y,x] = modeUpdate.copy() #
            mode[:,y,x] = modeUpdate
        endtime = time.time()
        print("{:.3f} sec".format((endtime-starttime)))
    endtimeTotal = time.time()
    print("elapsed time : {:.3f} sec".format((endtimeTotal-starttimeTotal)))
    return mode

def meanShiftVis(img,modes):
    '''
    '''
    if len(img.shape)==3:
        imgShape = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape)==2:
        imgShape = img
    height,width = imgShape.shape
    meanShiftImg = np.zeros(shape=img.shape,dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            meanShiftImg[y,x] = modes[:3,y,x]
    meanShiftImg = cv2.cvtColor(meanShiftImg, cv2.COLOR_Luv2BGR)
    return meanShiftImg

def main():
    # img0 = cv2.imread('/home/hyh/robot_vision/cap_img/22.png', cv2.IMREAD_COLOR)
    # img1 = cv2.imread('/home/hyh/robot_vision/cap_img/23.png', cv2.IMREAD_COLOR)
    img1 = img0

    img0 = imgResize(img0,rate=0.05)
    img1 = imgResize(img1,rate=0.05)

    modes = meanShift(img0,bandwidth=25)
    meanShiftImg = meanShiftVis(img0,modes)

    cv2.imshow('original image0',img0)
    cv2.imshow('mean shift image',meanShiftImg)
    cv2.waitKey(0)

if __name__=='__main__':
    main()
