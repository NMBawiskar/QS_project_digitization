import math
import cv2
import numpy as np
from scipy import ndimage
import os
import matplotlib.pyplot as plt
import random


def __rotate_img(image, page_no):
    edges = cv2.Canny(image, 400, 500, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    main_lines = []

    display_img = image.copy()
    display_img = cv2.line(display_img, (0, 300), (image.shape[0], 300), (0, 0, 255), 2)
    display_img = cv2.line(display_img, (0, 1800), (image.shape[0], 1800), (0, 0, 255), 2)

    median_angle = 0
    angles = []
    if lines is not None:
        for i in range(0, len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if -70 > angle > -90:
                    angle = 90 + angle
                elif 70 < angle < 90:
                    angle = angle - 90
                if 1800 > y1 > 300 and 1800 > y2 > 300 and 0 < abs(angle) < 35:
                    angles.append(angle)
                    display_img = cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    main_lines.append((x1, y1, x2, y2, angle))

                    if len(angles) >= 50:
                        break

            # if i % 100 == 0:
            #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
            #     cv2.resizeWindow('image', 750, 900)
            #     cv2.imshow("image", display_img)  # Show image
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
        mean = (np.mean(angles)) if len(angles) > 0 else 0
        std = np.std(angles) if len(angles) > 0 else 0

        # print(std, mean)
        # if std > mean:
        #     std = std / 2

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.distplot(angles, hist=True, kde=True,
        #              bins=int(180 / 5), color='darkblue',
        #              hist_kws={'edgecolor': 'darkblue'},
        #              kde_kws={'linewidth': 1})
        # sns.distplot([mean - std, mean + std], hist=True, kde=True,
        #              bins=int(180 / 5), color='green',
        #              hist_kws={'edgecolor': 'green'},
        #              kde_kws={'linewidth': 1})
        # plt.show()

        if len(angles) > 0:
            min_angle, max_angle = mean - std, mean + std
            considerable_lines = [line for line in main_lines if max_angle >= line[4] >= min_angle]
            # print(len([line[4] for line in considerable_lines]))
            # print(len([line[4] for line in considerable_lines if line[4] >= 0]))
            # print(len([line[4] for line in considerable_lines if line[4] <= 0]))
            median_angle = (np.mean([line[4] for line in considerable_lines])) if len(considerable_lines) > 0 else 0
        # print(median_angle)
        if median_angle < -45:
            median_angle = 90 + median_angle
        print(f"Angle of rotation for page {page_no} is {round(median_angle, 2)}\N{DEGREE SIGN}.")
        img_rotated = ndimage.rotate(image, median_angle)
    else:
        img_rotated = image.copy()
    return img_rotated, median_angle


def find_files_recursively(dir_path):
    files_to_process = list()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if not root.endswith("_ocr"):
                files_to_process.append(os.path.join(root, file))
    return files_to_process


path = r"Reports\extracted_png"
images = find_files_recursively(path)
print(f"Found {len(images)} images")
random.shuffle(images)
for img in images:
    print(img)

    img_obj = cv2.imread(img)
    rot_img_obj, angle = __rotate_img(img_obj, 0)

    # cv2.imshow('Original', img_obj)
    # cv2.imshow('Rotated', rot_img_obj)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if abs(angle) >= 1:
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img_obj)
        axarr[0].grid()
        axarr[1].imshow(rot_img_obj)
        axarr[1].grid()

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
