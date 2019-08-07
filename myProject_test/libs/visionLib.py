import cv2,time
import numpy as np

def keyX(p):
    """ used for sorted by x axis
    """
    return p[0]
def keyY(p):
    """ used for sorted by y axis
    """
    return p[1]
def arrContour2ListPoints(cnt):
    """ convert ndarray points to list points
    """
    lstPoitns = [[c[0],c[1]] for c in cnt]
    return lstPoitns
def distance(p,q):
    """ return distance of 2 point
    """
    x1,y1 = p
    x2,y2 = q
    return np.sqrt((x1-x2)**2+(y1-y2)**2)
def getCentroid(cnt):
    """ return Centroid of contours
    """
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx,cy)
    return centroid

# use moments for image with black background
def getSkew(cnt):
    """ return skew of contours
    """
    m = cv2.moments(cnt)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return 0
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    return skew

def deSkew(mat,skew,sz=1,v=(0,0)):
    """ return output after de skew.
        transform to v 
        sz : output scale
        skew : skew of mat"""
    rows,cols = mat.shape[:2]
    M = np.float32([[1,skew,v[0]], [0,sz,v[1]]])
    img = cv2.warpAffine(mat,M,(cols,rows), flags= cv2.INTER_LINEAR) #
    return img

def rotated(mat,I,angle,sz=1):
    """ return output after rotated.
        I : cennter 
        sz : output scale
        angle : rotated angle"""
    rows,cols = mat.shape[:2]
    M = cv2.getRotationMatrix2D((I[0],I[1]),angle,sz)
    dst = cv2.warpAffine(mat,M,(cols,rows))
    return dst

def gray2bgr(gray):
    """ convert gray image to bgr iamge
    """
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

def bgr2gray(bgr):
    """ convert bgr image to gray iamge
    """
    return cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

 
# mat = cv2.imread("D:/GitHub/visionLib/image/opencv.jpg")
# # out = deSkew(mat,0.5,centroid=(0,0))
# out = rotated(mat,10)

# stack = np.hstack((mat,out))

# cv2.imshow("",stack)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    





