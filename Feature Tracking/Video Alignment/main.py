from FrameStabilizer import FrameStabilizer
#import VideoStabilizer

if __name__ == '__main__':
    frameStabilizer = FrameStabilizer("input6.mp4",
                                      "H:/study/CV/Project-2",
                                      "output6-new-2000-015.mp4",
                                      "H:/study/CV/Project-2")
    frameStabilizer.stabilizeFrame()
    # stabilizedVideo = VideoStabilizer.stabilize_video("input3.mp4","output.mp4")
    print("Video Stabilized!")