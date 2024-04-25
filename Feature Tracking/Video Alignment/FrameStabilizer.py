import cv2
import numpy as np
import os
from FrameRotater import frameRotater
from FrameRotater import X_Y_shift_and_rotate
import ECC

class FrameStabilizer:

    def __init__(self, inputVideoPath, inputFramePath, outputVideoPath, outputFramePath):
        self.inputVideoPath = inputVideoPath
        self.inputFramePath = inputFramePath
        cap = cv2.VideoCapture(self.inputVideoPath)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            raise IOError("Error opening video file at {}".format(self.inputVideoPath))
        frame_index = 0
        os.chdir(inputFramePath)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"frame_{frame_index:04d}.png"
            cv2.imwrite(frame_filename, frame)
            frame_index += 1
        cap.release()
        self.outputFramePath = outputFramePath
        self.outputVideoPath = outputVideoPath
        self.frames = [fname for fname in os.listdir(inputFramePath) if fname.endswith(".png")]

    def stabilizeFrame(self):
        self.__applyFrameRotation()
        self.__imageToVideo(60)

    def __applyFrameRotation(self):
        frameIndex = 0
        os.chdir(self.inputFramePath)
        prevFrame = self.frames[0]
        prevImg = cv2.imread(prevFrame)
        os.chdir(self.outputFramePath)
        cv2.imwrite(f"frame_{frameIndex:04d}.png", prevImg)
        for i in range(1,len(self.frames)):
            currFrame = self.frames[i]
            prevFrame = self.frames[0]
            firstFrame = self.frames[0]
            #frameRotater(currFrame,prevFrame,self.inputFramePath,self.outputFramePath,frameIndex)
            X_Y_shift_and_rotate(prevFrame,currFrame,self.inputFramePath,self.outputFramePath,frameIndex)
            #ECC.frameRotater_ECC(currFrame, prevFrame, self.inputFramePath, self.outputFramePath, frameIndex)
            frameIndex += 1

    # def __estimateMotion(self):
    #     motions = []
    #     for i in range(1, len(self.grayFrames)):
    #         p0 = cv2.goodFeaturesToTrack(self.grayFrames[i-1], maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    #         p1, st, err = cv2.calcOpticalFlowPyrLK(self.grayFrames[i-1], self.grayFrames[i], p0, None)
    #         goodOld = p0[(st == 1) & (err < 50)]
    #         goodNew = p1[(st == 1) & (err < 50)]
    #
    #         print(goodOld)
    #         print(goodNew)
    #         #matrix, _ = cv2.findHomography(goodOld, goodNew, cv2.RANSAC, 5.0)
    #         matrix, inliers = cv2.estimateAffinePartial2D(goodOld,goodNew)
    #         if matrix is not None:
    #             matrix = np.vstack([matrix, [0,0,1]])
    #         motions.append(matrix)
    #     return motions
    #
    # def __smoothTradjectory(self, motions):
    #     smoothedMotions = [np.eye(3)] * len(motions)
    #     for i in range(1,len(motions)-1):
    #         # if i == 0 or i == len(motions) - 1:
    #         #     smoothedMotions.append(motions[i])
    #         # else:
    #         smoothedMotions[i] = np.mean(np.array([motions[i-1], motions[i], motions[i+1]]), axis=0)
    #         # smoothedMotions.append(smoothedMat)
    #     return smoothedMotions
    #
    # def __applyTransformation(self, smoothedMotions):
    #     stabilizedFrames = []
    #     for frame, motion in zip(self.frames, smoothedMotions):
    #         height, width = frame.shape[:2]
    #         stabilizedFrame = cv2.warpPerspective(frame,motion,(width,height))
    #         stabilizedFrames.append(stabilizedFrame)
    #     return stabilizedFrames

    def __imageToVideo(self, fps):
        images = [fname for fname in os.listdir(self.outputFramePath) if fname.endswith(".png")]
        frame = cv2.imread(os.path.join(self.outputFramePath, images[0]))
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.outputVideoPath,fourcc,self.fps,(width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(self.outputFramePath, image)))
        video.release()