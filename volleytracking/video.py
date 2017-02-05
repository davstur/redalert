
import numpy as np
import cv2
import matplotlib
matplotlib.use("Qt5Agg") # before importing pyplot
import matplotlib.pyplot as plt


filenameIn = "150107_Schoenenwerd-Jona.mp4"

def backgroundSubtractor(filenameIn):
  videoCapture = cv2.VideoCapture(filenameIn)

  fgbg = cv2.createBackgroundSubtractorMOG2()
  while True:
      ret, frame = videoCapture.read()
      fgmask = fgbg.apply(frame)
      cv2.imshow('frame', fgmask)


      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
  videoCapture.release()
  cv2.destroyAllWindows()



def denseOpticalFlow(filenameIn):
  videoCapture = cv2.VideoCapture(filenameIn)
  ret, frame1 = videoCapture.read()
  prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[...,1] = 255
  while(1):
      ret, frame2 = videoCapture.read()
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
      flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = ang*180/np.pi/2
      hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
      bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
      cv2.imshow('frame2',bgr)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
      elif k == ord('s'):
          cv2.imwrite('opticalfb.png',frame2)
          cv2.imwrite('opticalhsv.png',bgr)
      prvs = next
  videoCapture.release()
  cv2.destroyAllWindows()


def cutToFrames(filenameIn, frameStart, frameEnd):

  filenameOut = "{}_cut_{}-{}.mp4".format(filenameIn, frameStart, frameEnd)

  videoCapture = cv2.VideoCapture(filenameIn)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  print("Frames per second: {0}".format(fps))
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  videoWriter = None

  cnt = 0
  while cnt <= frameEnd:

    (_, frame) = videoCapture.read()

    if videoWriter is None:
      # store the image dimensions, initialzie the video writer,
      # and construct the zeros array
      (h, w) = frame.shape[:2]
      videoWriter = cv2.VideoWriter(filenameOut, fourcc, fps, (w, h), True)

    if (cnt >= frameStart) & (cnt <= frameEnd):
      videoWriter.write(frame)

    cnt += 1

  print("[INFO] cleaning up...")
  cv2.destroyAllWindows()
  videoWriter.release()

# cutToFrames(filenameIn, 100, 1000)



def outputFramesToCut(filenameIn):

  cnt = 0
  videoCapture = cv2.VideoCapture(filenameIn)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)

  frameCnts = []
  stopCnt = 0
  while stopCnt < 2:

    (_, frame) = videoCapture.read()
    t = cnt / fps
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(10)  # returns -1 if nothing is pressed
    if k==32: # spacebar
      print("cnt: {}, t: {}".format(cnt, t))
      frameCnts.append(cnt)
      stopCnt += 1

    cnt += 1

  return frameCnts


def labelFrames(filenameIn):

  cnt = 0
  videoCapture = cv2.VideoCapture(filenameIn)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)

  labels = []
  stop = False
  while not stop:

    (_, frame) = videoCapture.read()

    if frame is None: break

    t = cnt / fps
    cv2.imshow("Frame", frame)

    if (cnt % 25) == 0:

      k = cv2.waitKey(0)  # returns -1 if nothing is pressed

      if k==ord("a"): # when playing
        labels.append((cnt, 1))
      elif k==ord("b"): # when playing
        labels.append((cnt, 0))
      elif k==ord("q"):
        stop = True

      print("cnt: {}, t: {}".format(cnt, t))
      print(labels)

    cnt += 1

  print(labels)


def computeDifferences(filenameIn):

  cnt = 0
  videoCapture = cv2.VideoCapture(filenameIn)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  differences = [(0, 0)]

  prevFrame = None
  k = 100
  kernel = np.ones((k, k), np.float32)/(k*k)
  fgbg = cv2.createBackgroundSubtractorMOG2()
  print("Using createBackgroundSubtractorMOG2")

  while cnt < 1500:

    t = cnt / fps
    print("cnt: {}, t: {}".format(cnt, t))

    (_, frame) = videoCapture.read()
    if frame is None: break
    if (cnt % 25) != 0:
      cnt += 1
      continue

    # frame = cv2.filter2D(frame, -1, kernel)
    frame = fgbg.apply(frame)
    # cv2.imshow("Frame", frame)
    # cv2.waitKey(1)

    if prevFrame is None: prevFrame = frame

    if cnt > 0:
      d = sum(abs((frame - prevFrame).flatten()))
      differences.append((cnt, d))
      prevFrame = frame

    cnt += 1

  videoCapture.release()
  cv2.destroyAllWindows()

  return differences


# cutToFrames(filenameIn, 5800, 5800 + 25 * 60 * 2)
filenameIn = "150107_Schoenenwerd-Jona.mp4_cut_5800-8800.mp4"
# labelFrames(filenameIn)

# backgroundSubtractor(filenameIn)
denseOpticalFlow(filenameIn)

exit()


differences = np.array(computeDifferences(filenameIn))
print(differences)

labels = np.array([(0, 0), (25, 0), (50, 0), (75, 0), (100, 1), (125, 1), (150, 1), (175, 1), (200, 1), (225, 1), (250, 1), (275, 1), (300, 1), (325, 0), (350, 0), (375, 0), (400, 0), (425, 0), (450, 0), (475, 0), (500, 0), (525, 0), (550, 0), (575, 0), (600, 0), (625, 0), (650, 0), (675, 0), (700, 0), (725, 0), (750, 0), (775, 0), (800, 0), (825, 0), (850, 1), (875, 1), (900, 1), (925, 1), (950, 1), (975, 1), (1000, 1), (1025, 1), (1050, 1), (1075, 1), (1100, 0), (1125, 0), (1150, 0), (1175, 0), (1200, 0), (1225, 0), (1250, 0), (1275, 0), (1300, 0), (1325, 0), (1350, 0), (1375, 0), (1400, 0), (1425, 0), (1450, 0), (1475, 0), (1500, 0), (1525, 0), (1550, 0), (1575, 0), (1600, 0), (1625, 0), (1650, 0), (1675, 0), (1700, 0), (1725, 0), (1750, 1), (1775, 1), (1800, 1), (1825, 1), (1850, 1), (1875, 1), (1900, 1), (1925, 1), (1950, 1), (1975, 1), (2000, 1), (2025, 1), (2050, 1), (2075, 1), (2100, 1), (2125, 1), (2150, 0), (2175, 0), (2200, 0), (2225, 0), (2250, 0), (2275, 0), (2300, 0), (2325, 0), (2350, 0), (2375, 0), (2400, 0), (2425, 0), (2450, 0), (2475, 0), (2500, 0), (2525, 0), (2550, 0), (2575, 0), (2600, 0), (2625, 0), (2650, 0), (2675, 1), (2700, 1), (2725, 1), (2750, 1), (2775, 1), (2800, 1), (2825, 0), (2850, 0), (2875, 0), (2900, 0), (2925, 0), (2950, 0), (2975, 0), (3000, 0)])
f, axarr = plt.subplots(2, 1, sharex=True)
axarr[0].plot(labels[:, 0], labels[:, 1])
axarr[0].set_title('labels')
axarr[1].plot(differences[:, 0], differences[:, 1])
axarr[1].set_title('differences')
plt.draw()
plt.gcf().canvas.manager.window.raise_()
plt.show()





