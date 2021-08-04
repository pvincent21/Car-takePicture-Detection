#!/usr/bin/python3
import cv2
import numpy as np
import time
import os
import sys
import shutil
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
from postdata import postimage, getlastnumber

# Ambil data id_mobil terakhir
id_mobil = 1 + int(getlastnumber())
print("Sekarang " + str(id_mobil))
status = 0

ycor = 200
offset = 6
count = 0
poto = 0

detect = []

folderFoto = 'Foto'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def center_rec(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def main():
    global mulai, count, poto, large, status

    #video = cv2.VideoCapture(0)
    video = cv2.VideoCapture('Video1.mp4')
#    video = VideoStream(src=0,usePiCamera=True,resolution=(464,368)).start()

    net = cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

    fps = FPS().start()

    while True:
        ret, frame = video.read()
        frame1 = frame.copy()
        frame = cv2.resize(frame, (464, 368))
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detection = net.forward()

        for i in np.arange(0, detection.shape[2]):
            confidence = detection[0, 0, i, 2]
            if confidence > 0.5:
                id = detection[0, 0, i, 1]
                print(id)
                if status == 0:
                    if id == 15:
                        box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')
                        cv2.rectangle(frame, (startX, startY),
                                      (endX, endY), (0, 255, 0))

                        center = center_rec(
                            startX, startY, (endX-startX), (endY-startY))
                        detect.append(center)

                        large = ((endX-startX)*(endY-startY))/250
                        print(large)

                        if large > 30:
                            count += 1
                            poto += 1
                            lokasi = "Foto/car" + str(count) + ".jpg"

                            key = cv2.waitKey(100) & 0xff
                            if key == 10:
                                break
                            elif poto >= 9:
                                try:
                                    keterangan = "Masuk"
                                    postimage(id_mobil, keterangan, lokasi)
                                    status = 1
                                    count = 0
                                    poto = 0
                                    first = time.time()
                                    break
                                except Exception as e:
                                    print("keterangan")
                                    print(e)
                                    break
                            cv2.imwrite(lokasi, frame1)
                ###############################################
                elif status == 1:
                    last = time.time()
                    selisih = last - first
                    selisihm = int(selisih)
                    if selisihm == 5:  # update waktu / 15 menit  #####
                        first = time.time()
                        lokasi = "Foto/car.jpg"
                        cv2.imwrite(lokasi, frame1)
                        keterangan = "Check"
                        postimage(id_mobil, keterangan, lokasi)
                        time.sleep(2.0)
                ##############################################
                    if id == 15:
                        box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')
                        cv2.rectangle(frame, (startX, startY),
                                      (endX, endY), (0, 255, 0))

                        center = center_rec(
                            startX, startY, (endX-startX), (endY-startY))
                        detect.append(center)
                        large = ((endX-startX)*(endY-startY))/250

                        if large < 30:
                            count += 1
                            poto += 1
                            lokasi = "Foto/car" + str(count) + ".jpg"
                            key = cv2.waitKey(100) & 0xff
                            if key == 10:
                                break
                            elif poto >= 13:
                                try:
                                    keterangan = "keluar"
                                    postimage(id_mobil, keterangan, lokasi)
                                    id_mobil = id_mobil + 1
                                    status = 0
                                    count = 0
                                    poto = 0
                                    break
                                except:
                                    print("keterangan")
                                    break
                            cv2.imwrite(lokasi, frame1)

        print("Car Count : " + str(count))
        cv2.imshow("Frame", frame)
    #    time.sleep(1/100)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#video.release()
    cv2.destroyAllWindows()

######################################################################################################################################################################3
if __name__ == "__main__":
    main()
