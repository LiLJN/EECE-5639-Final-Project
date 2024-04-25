import cv2
import numpy as np
import matplotlib.pyplot as plt

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    _, prev = cap.read()

    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevCornerFeatures = cv2.goodFeaturesToTrack(prevGray, maxCorners=200, qualityLevel=0.01, minDistance=30,
                                                 blockSize=3)
    #print("Next")
    originalX = [prevCornerFeatures[0][0][0]]
    originalY = [prevCornerFeatures[0][0][1]]
    newX = []
    newY = []
    transforms = []

    while True:
        success, curr = cap.read()
        if not success:
            break
        currGray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        badCurrCornerFeatures = cv2.goodFeaturesToTrack(currGray, maxCorners=200, qualityLevel=0.01, minDistance=30,
                                                            blockSize=3)
        originalX.append(badCurrCornerFeatures[0][0][0])
        originalY.append(badCurrCornerFeatures[0][0][1])

        currCornerFeatures, status, _ = cv2.calcOpticalFlowPyrLK(prevGray, currGray, prevCornerFeatures, None)
        goodPrevCornerFeatures = prevCornerFeatures[status == 1]
        goodCurrCornerFeatures = currCornerFeatures[status == 1]
        tran, _ = cv2.estimateAffinePartial2D(goodPrevCornerFeatures, goodCurrCornerFeatures)

        dx = tran[0, 2]
        dy = tran[1, 2]
        newX.append(dx)
        newY.append(dy)
        transforms.append([dx, dy])

        prevGray = currGray.copy()
        prevCornerFeatures = goodCurrCornerFeatures.reshape(-1, 1, 2)

    cumulative_transform = np.cumsum(transforms, axis=0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(len(transforms)):
        success, frame = cap.read()
        if not success:
            break

        dx, dy = cumulative_transform[i]

        transformationMatrix = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        stabilizedFrame = cv2.warpAffine(frame, transformationMatrix, size)

        out.write(stabilizedFrame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Plotting
    t = []
    for i in range(len(originalX)):
        t.append(i)
    newT = []
    for i in range(len(newX)):
        newT.append(i)

    plt.figure(1)
    fig1 = plt.subplot(121)
    plt.plot(t, originalX, lw=1, color='blue')
    fig1.set_title("Original trajectory on x-axis")
    fig2 = plt.subplot(122)
    plt.plot(newT, newX, lw=1, color='red')
    fig2.set_title("New trajectory on x-axis")
    plt.show()

    plt.figure(2)
    fig3 = plt.subplot(121)
    plt.plot(t, originalY, lw=1, color = 'blue')
    fig3.set_title("Original trajectory on y-axis")
    fig4 = plt.subplot(122)
    plt.plot(newT, newY, lw=1, color = 'red')
    fig4.set_title("New trajectory on y-axis")
    plt.show()