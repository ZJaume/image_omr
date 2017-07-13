import cv2
import sys

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 3
maxLineGap = 4

lines = cv2.HoughLines(edges,1,np.pi/180,1)
for line in lines:
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(out,(x1,y1),(x2,y2),(255,255,255),2)

lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(out,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite("dilated.png",out)
