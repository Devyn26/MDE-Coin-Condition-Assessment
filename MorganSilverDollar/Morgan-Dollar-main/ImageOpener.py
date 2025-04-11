"""Unused"""
# import cv2
# import os
#
#
# def loadImages(mode):
#     # Replace ./ with path to the cloned GitHub Repository.
#     imageFolderPath = "ScrapedImages/"
#     # Find the number of files (images) in image folder
#     path, dirs, files = next(os.walk(imageFolderPath))
#     file_count = len(files)
#     print("Images Found: ", file_count)
#
#     images = []
#     imgNames = []
#
#     # For every single image in the folder, append image name to imageFolderPath
#     # Then, read and open the image.
#     i = 0
#     for img in files:
#         file_path = os.path.abspath(imageFolderPath + img)
#         if os.path.isfile(file_path):
#             print(path+img)
#             if mode == 'color':
#                 images.append(cv2.imread(file_path, cv2.IMREAD_COLOR))
#             elif mode == 'grey':
#                 images.append(cv2.imread(file_path))
#             imgNames.append(img)
#             # windowName = "from David Lawrence Rare Coins " + img
#             # cv2.imshow(windowName, images[i])
#             i += 1
#
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     return images, imgNames
#
#
# if __name__ == '__main__':
#     loadImages('color')
