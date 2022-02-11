# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import cv2
# import math
# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image
# def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
#     line_img = np.zeros(
#         (
#             img.shape[0],
#             img.shape[1],
#             3
#         ),
#         dtype=np.uint8
#     )
#     img = np.copy(img)
#     if lines is None:
#         return
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
#     img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
#     return img
# def pipeline(image):
#     """
#     An image processing pipeline which will output
#     an image with the lane lines annotated.
#     """
#     height = image.shape[0]
#     width = image.shape[1]
#     region_of_interest_vertices = [
#         (0, height),
#         (width / 2, height / 2),
#         (width, height),
#     ]
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     cannyed_image = cv2.Canny(gray_image, 100, 200)
 
#     cropped_image = region_of_interest(
#         cannyed_image,
#         np.array(
#             [region_of_interest_vertices],
#             np.int32
#         ),
#     )
 
#     lines = cv2.HoughLinesP(
#         cropped_image,
#         rho=6,
#         theta=np.pi / 60,
#         threshold=160,
#         lines=np.array([]),
#         minLineLength=40,
#         maxLineGap=25
#     )
 
#     left_line_x = []
#     left_line_y = []
#     right_line_x = []
#     right_line_y = []
 
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             slope = (y2 - y1) / (x2 - x1)
#         if math.fabs(slope) < 0.5:
#             continue
#         if slope <= 0:
#             left_line_x.extend([x1, x2])
#             left_line_y.extend([y1, y2])
#         else:
#             right_line_x.extend([x1, x2])
#             right_line_y.extend([y1, y2])
#         min_y = int(image.shape[0] * (3 / 5))
#         max_y = int(image.shape[0])
#         poly_left = np.poly1d(np.polyfit(
#             left_line_y,
#             left_line_x,
#             deg=1
#         ))
    
#         left_x_start = int(poly_left(max_y))
#         left_x_end = int(poly_left(min_y))
    
#         poly_right = np.poly1d(np.polyfit(
#             right_line_y,
#             right_line_x,
#         deg=1
#         ))
    
#         right_x_start = int(poly_right(max_y))
#         right_x_end = int(poly_right(min_y))
#         line_image = draw_lines(
#             image,
#             [[
#                 [left_x_start, max_y, left_x_end, min_y],
#                 [right_x_start, max_y, right_x_end, min_y],
#             ]],
#             thickness=5,
#         )
#     return line_image
import sys
import math
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = (r'E:\DRIVE\AUTOZ\ex1_lane.jpg')
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    blur=cv.GaussianBlur(cdst, (5, 5), 0)
    cv.imshow('BLURREd',blur)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])