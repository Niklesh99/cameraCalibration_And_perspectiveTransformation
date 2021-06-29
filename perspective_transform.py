import cv2 as cv
import numpy as np
import sys

def randomColor():
    color = np.random.randint(0, 255,(1, 3))
    return color[0].tolist()

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []


img1 = cv.imread("calib_15.png")
img2 = cv.imread('calib_4.png')

cv.imshow('Before Translation', img1)
cv.waitKey(0)


    # [find-corners]
ret1, corners1 = cv.findChessboardCorners(img1, (9,6))
ret2, corners2 = cv.findChessboardCorners(img2,  (9,6))
    # [find-corners]

if not ret1 or not ret2:
    print("Error, cannot find the chessboard corners in both images.")
    sys.exit(-1)

    # [estimate-homography]
H, _ = cv.findHomography(corners1, corners2)
print(H)
print(_)
    # [estimate-homography]

    # [warp-chessboard]
img1_warp = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    # [warp-chessboard]

img_draw_warp = cv.hconcat([img2, img1_warp])
cv.imshow("After Translation", img_draw_warp )
cv.waitKey(0)


# img_draw_matches = cv.hconcat([img1, img2])
# for i in range(len(corners1)):
#     pt1 = np.array([corners1[i][0], corners1[i][1], 1])
#     pt1 = pt1.reshape(3, 1)
#     pt2 = np.dot(H, pt1)
#     pt2 = pt2/pt2[2]
#     end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
#     cv.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, randomColor(), 2)

# cv.imshow("Draw matches", img_draw_matches)
# cv.waitKey(0)
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# rmat = np.empty((3,3))

# cv.Rodrigues([[0.16852249],[0.27488958],[0.01355863]], rmat)
# print(rmat)


# cv.solvePnPRansac()
# corners1 = corners1.reshape(corners1.shape[0],corners1.shape[2])
# corners2 = corners2.reshape(corners2.shape[0],corners2.shape[2])


# if ret == True:
#     objpoints.append(objp)

#     corners1 = cv.cornerSubPix(image1,corners, (11,11), (-1,-1), criteria)
#     imgpoints1.append(corners1)

#         # # Draw and display the corners
#         # cv2.drawChessboardCorners(img, (7,6), corners2, ret)
#         # smaller = cv2.resize(img, (1024,768))
        
#     else:
#         print("Not running")

#     cv2.destroyAllWindows()

# img_draw_matches = cv.hconcat([img1, img2])
# for i in range(len(corners1)):
#     pt1 = np.array([corners1[i][0], corners1[i][1], 1])
#     pt1 = pt1.reshape(3, 1)
#     pt2 = np.dot(H, pt1)
#     pt2 = pt2/pt2[2]
#     end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
#     cv.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, randomColor(), 2)

# cv.imshow("Draw matches", img_draw_matches)
# cv.waitKey(0)



