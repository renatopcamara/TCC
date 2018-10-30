
# coding: utf-8


import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('barra2.mp4')
path = "C:\\Users\\renato\\Desktop\\frames\\"
success,image = vidcap.read()
count = 0
success,image = vidcap.read()
success = True
while success:
    file = "frame" + str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg"
    filename = path + file
    cv2.imwrite(filename, image)     # save frame as JPEG file
    success,image = vidcap.read()
    vidcap.set(1,count*10)
    print ('Read a new frame: ', success)
    count += 1
    if count == 10:
        break

