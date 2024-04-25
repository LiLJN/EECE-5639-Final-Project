from FrameStabilizer import FrameStabilizer
import VideoStabilizer

if __name__ == '__main__':
    # frameStabilizer = FrameStabilizer("input3.mp4",
    #                                   "/Users/alanchen15/Desktop/Alan_NEU/EECE/EECE 5639/Final Project/Final Project/inputFrame",
    #                                   "output.mp4",
    #                                   "/Users/alanchen15/Desktop/Alan_NEU/EECE/EECE 5639/Final Project/Final Project/outputFrame")
    # frameStabilizer.stabilizeFrame()
    stabilizedVideo = VideoStabilizer.stabilize_video("input3.mp4","output.mp4")
    print("Video Stabilized!")