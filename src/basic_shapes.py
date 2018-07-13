import cv2
import numpy as np

def circles(canvas, centers, max_rad=256, gap=5, thick=2, color=(0,0,0)):
    for center in centers:
        for rad in range(1,max_rad,gap):
            cv2.circle(canvas,center,rad,color,thick)

def radialShape(canvas, center, radius, count, color=(0,0,0), thick=1):
    gap_theta = np.linspace(0, np.pi * 2, count)
    offsets = [polar2cart(radius, theta) for theta in gap_theta]
    outmosts = [(center[0]+int(off_x), center[1]+int(off_y)) for (off_x, off_y) in offsets]
    for out in outmosts:
        cv2.line(canvas, center, out, color, thickness=thick)
