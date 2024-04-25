import os

import numpy as np
import cv2


def frameRotater(imgPath1, imgPath2,inputFramePath,outputFramePath,frameIndex):
    os.chdir(inputFramePath)
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #gray1 = cv2.GaussianBlur(gray1, (3, 3), 1, 1)
    #gray2 = cv2.GaussianBlur(gray2, (3, 3), 1, 1)

    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    height, width = gray1.shape
    img3 = cv2.warpPerspective(img1, h, (width, height))

    for i in range(height):
        for j in range(width):
            for k in range(3):
                if img3[i][j][k] == 0:
                    img3[i][j][k] = img2[i][j][k]

    os.chdir(outputFramePath)
    frameFileName = f"frame_{frameIndex:04d}.png"
    cv2.imwrite(frameFileName, img3)




def X_Y_shift_and_rotate(fig_ref, fig_current, inputFramePath, outputFramePath, frameIndex):
    ##########gray1  current gray2 prev
    '''
    img1 = cv2.imread("H:/study/CV/Project/IMG4.JPG")# current
    img2 = cv2.imread("H:/study/CV/Project/IMG5.JPG")# prev reference
    '''
    os.chdir(inputFramePath)
    img1 = cv2.imread(fig_ref)
    img2 = cv2.imread(fig_current)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = gray1.shape[:2]
    template = gray1[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]

    res = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(max_val)

    if max_val > 0.95:
        x_shift = int(w * 0.4) - max_loc[0];
        y_shift = int(h * 0.4) - max_loc[1];
        # plt.imshow(gray1, 'gray'),plt.show()
        # x_shift, y_shift=X_Y_shift(gray2, gray1)
        height, width, channel = img1.shape
        mat_translation = np.float32([[1, 0, -x_shift], [0, 1, -y_shift]])
        img3 = cv2.warpAffine(img1, mat_translation, (width, height))
    else:
        orb = cv2.ORB_create(2000)

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)

        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        numGoodMatches = int(len(matches) * 0.15)
        matches = matches[:numGoodMatches]

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        height, width = gray1.shape
        img3 = cv2.warpPerspective(img1, h, (width, height))

    os.chdir(outputFramePath)
    frameFileName = f"frame_{frameIndex:04d}.png"
    cv2.imwrite(frameFileName, img3)