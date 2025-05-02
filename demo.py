"""
demo.py

Code for running the prototype for the critical design review presentation.

Author: Eric Morley
Date: 5/2/2025
"""

import cv2
import prepare_coin_image
from LincolnCent import GUI
from MorganSilverDollar.Morgan_Dollar_main import InputCoin

if __name__ == '__main__':
    Lincoln_obv_img_path = 'test_images/Lincoln_test_obv.jpg'
    Lincoln_rev_img_path = 'test_images/Lincoln_test_rev.jpg'

    MSD_obv_img_path = 'test_images/MSD_test_obv.jpg'
    MSD_rev_img_path = 'test_images/MSD_test_rev.jpg'

    Lincoln_obv_img = prepare_coin_image.process(Lincoln_obv_img_path)
    Lincoln_rev_img = prepare_coin_image.process(Lincoln_rev_img_path, hough_param2=55)

    MSD_obv_img = prepare_coin_image.process(MSD_obv_img_path)
    MSD_rev_img = prepare_coin_image.process(MSD_rev_img_path, hough_minRadius=310, hough_maxRadius=330)

    # save to show sample processed images
    cv2.imwrite('test_images/Lincoln_obv_img_proc.jpg', Lincoln_obv_img)
    cv2.imwrite('test_images/Lincoln_rev_img_proc.jpg', Lincoln_rev_img)

    cv2.imwrite('test_images/MSD_obv_img_proc.jpg', Lincoln_obv_img)
    cv2.imwrite('test_images/MSD_rev_img_proc.jpg', Lincoln_rev_img)

    # Morgan Silver Dollar Demo
    InputCoin.runMSDCode(MSD_obv_img, MSD_rev_img)

    # Lincoln Wheat Cent Demo
    GUI.runLWCCode()