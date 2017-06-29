from orb_matcher import*
import cv2
import numpy as np
from matplotlib import pyplot as plt

def homographyComputer(imgs):

    homMatrices = []
    for i in reversed(range(len(imgs) - 1)):

        '''
        if i == 0:
            temp = imgs[i]
            imgs[i] = imgs[i+1]
            imgs[i+1] = temp
        '''
        #print i

        #returns matches (DMatch objects) where first
        (matches, key1, key2) = keypointMatcher(imgs[i+1], imgs[i])
        src_points=np.float32([key1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_points=np.float32([key2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        #gives a matrix which when applied to img1, gives an allignment similar to img2
        if i == 0:
            M,mask=cv2.findHomography(dst_points,src_points,cv2.RANSAC,5.0)
            result = warpTwoImages(imgs[i+1], imgs[i], M)

        M,mask=cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
        result = warpTwoImages(imgs[i], imgs[i+1], M)
        imgs[i] = result
        plt.imshow(result, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        #print M
        #applies M on
        homMatrices.append(M)
        #out = cv2.warpPerspective(imgs[i+1], M, (imgs[i].shape[1]*2,imgs[i].shape[0]*2))
        #out = warpTwoImages(imgs[i], imgs[i+1], homMatrices[i])
        #out = np.add(out, imgs[i])

    #print out.shape
    '''
    plt.imshow(out, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    '''
    plt.subplot(1,3,3)
    plt.title("second image")
    plt.imshow(imgs[i+1], cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.subplot(1,3,2)
    plt.title("second image warped")
    plt.imshow(out, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


    plt.subplot(1,3,1)
    plt.title("first image")
    plt.imshow(imgs[i], cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    plt.imshow(out, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    return homMatrices


def homographyComputer2(imgs):
    (matches, key1, key2) = keypointMatcher(imgs[1], imgs[0])
    src_points=np.float32([key1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_points=np.float32([key2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
    img5 = warpTwoImages(imgs[0], imgs[1], M)

    plt.imshow(img5, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    (matches, key1, key2) = keypointMatcher(imgs[3], imgs[2])
    src_points=np.float32([key1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_points=np.float32([key2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
    img6 = warpTwoImages(imgs[2], imgs[3], M)

    plt.imshow(img6, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    (matches, key1, key2) = keypointMatcher(img6, img5)
    src_points=np.float32([key1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_points=np.float32([key2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    M,mask=cv2.findHomography(src_points,dst_points,cv2.RANSAC,5.0)
    img7 = warpTwoImages(img5, img6, M)

    plt.imshow(img7, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()



def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    #print pts1
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    wpts2 = cv2.perspectiveTransform(pts2, H)
    #print wpts2
    pts = np.concatenate((pts1, wpts2), axis=0)
    #print pts.min(axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel()- 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel()+0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    print Ht
    print H
    print Ht.dot(H)
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result
#def sticher(imgs):





img1 = cv2.imread("cover1.jpg")
#img1 = cv2.resize(img1,(img1.shape[1]/2, img1.shape[0]/2),cv2.INTER_AREA )
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#size = 296, 1000
#img3 = np.zeros(size, dtype=np.uint8)
#img3[:, 474:] = img1
#plt.imshow(img3, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#print img1.shape
img2 = cv2.imread("cover2.jpg")
#img2 = cv2.resize(img2,(img2.shape[1]/2, img2.shape[0]/2),cv2.INTER_AREA )

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread("cover3.jpg")
#img3 = cv2.resize(img3,(img3.shape[1]/2, img3.shape[0]/2),cv2.INTER_AREA )

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

img4 = cv2.imread("cover4.jpg")
#img4 = cv2.resize(img4,(img4.shape[1]/2, img4.shape[0]/2),cv2.INTER_AREA )

img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)


#print img2.shape
imgs = [img1, img2,img3, img4]

'''
for i in range(len(imgs)):
    #imgs[i] = cv2.medianBlur(imgs[i],3)
    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #imgs[i] = cv2.filter2D(imgs[i], -1, kernel)
    imgs[i] = cv2.resize(imgs[i],(int(imgs[i].shape[1]), int(imgs[i].shape[0])),cv2.INTER_AREA )
    #imgs[i] = cv2.pyrDown(imgs[i])
    #imgs[i] = cv2.medianBlur(imgs[i],3)
    #imgs[i] = cv2.medianBlur(imgs[i],3)
    print imgs[i].shape
'''

#homMatrices = homographyComputer(imgs)
homographyComputer(imgs)
