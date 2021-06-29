import numpy as np
import cv2 as cv
import glob

from numpy.core.fromnumeric import mean


chessboardSize = (9,6)
frameSize = (1440, 1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0]* chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calib_*.png')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)


    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs  = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print('\n CamerMatrix:  \n', cameraMatrix)
print('\n Disotion:  \n', dist)
print('\n rotaion Vector:  \n', rvecs)
print('\n Translation:  \n', tvecs)




img = cv.imread('calib_5.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))


dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)


mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print('\n total_error: {}'.format(mean_error/len(objpoints)))
print('\n\n\n')


####PERSPECTIVE TRANSFORMATION
#1. FIND HOMOGRAPHY
        # int indexArray[4] = {

        #                0,// upper left corner (0,0) // No. 0

        #                board_w - 1,// upper right corner (w-1,0) // 8

        #                (board_h - 1)*board_w,// Bottom left corner (0, h-1) // 5x9 = No. 45

        #                (board_h - 1)*board_w + board_w - 1// upper right corner (w-1, h-1) // 5 * 9 + 8 = 53

        #        };



K = newCameraMatrix
ret, rvec, tvec = cv.solvePnP(objpoints, imgpoints, corners2, K, dist)

R, _ = cv.Rodrigues(rvec)

# _,R_inv = cv.invert(R)
# _,K_inv = cv.invert(K)

# Hr = np.matmul(K, np.matmul(K_inv, R_inv))
# C = np.matmul(-R_inv,tvec)
# Cz = C[2]
# temp_vector = np.matmul(-K,C/Cz)

# for i, val in enumerate(temp_vector):
#     Ht[i][2] = val

# homography = np.matmul(Ht,Hr)
# warped_img =cv2.warpPerspective(img,homography,(img.shape[1],img.shape[0]))

