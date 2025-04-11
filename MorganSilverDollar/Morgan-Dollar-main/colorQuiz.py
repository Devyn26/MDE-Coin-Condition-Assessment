"""Deprecated"""
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# import pandas as pd
#
# df = pd.DataFrame(columns=['Image', 'Color'])
#
#
# class Interface:
#     def __init__(self, image):
#         self.index = 0
#         self.image = image
#
#     def lustrous(self, event):
#         temp = pd.DataFrame([[self.image, 'Lustrous Silver']],
#                             columns=['Image', 'Color'],)
#         global df
#         df = pd.concat([df, temp], ignore_index=True)
#         plt.close('all')
#
#     def silver(self, event):
#         temp = pd.DataFrame([[self.image, 'Silver']],
#                             columns=['Image', 'Color'])
#         global df
#         df = pd.concat([df, temp], ignore_index=True)
#         plt.close('all')
#
#     def dull(self, event):
#         temp = pd.DataFrame([[self.image, 'Dull Silver/Gray']],
#                             columns=['Image', 'Color'])
#         global df
#         df = pd.concat([df, temp], ignore_index=True)
#         plt.close('all')
#
#     def toned(self, event):
#         temp = pd.DataFrame([[self.image, 'Toned']],
#                             columns=['Image', 'Color'])
#         global df
#         df = pd.concat([df, temp], ignore_index=True)
#         plt.close('all')
#
#
# def main():
#
#     global df
#     dirname = os.path.abspath('Images/ColorTraining/') + '\\'
#     files = os.listdir(dirname)
#
#     for index, filename in enumerate(files):
#
#         interface = Interface(filename)
#
#         img = cv2.imread(dirname + filename)
#         RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         fig, ax = plt.subplots()
#         ax.imshow(RGB_img)
#         ax.axis('off')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title('Morgan Silver Dollar Classification\n WARNING: Close the Excel before starting')
#
#         lust = fig.add_axes([0.04, 0.03, 0.2, 0.075])
#         sil = fig.add_axes([0.28, 0.03, 0.2, 0.075])
#         dull = fig.add_axes([0.52, 0.03, 0.2, 0.075])
#         tone = fig.add_axes([0.76, 0.03, 0.2, 0.075])
#
#         l = Button(lust, 'Lustrous Silver')
#         l.on_clicked(interface.lustrous)
#         s = Button(sil, 'Silver')
#         s.on_clicked(interface.silver)
#         d = Button(dull, 'Dull Silver / Gray')
#         d.on_clicked(interface.dull)
#         t = Button(tone, 'Toned')
#         t.on_clicked(interface.toned)
#
#         plt.show()
#         # nameArr = filename.split()
#     writeDir = os.path.abspath('TrainingData/') + '\\'
#     df.to_excel(writeDir + 'colorTraining.xlsx')
#
#
# if __name__ == '__main__':
#     main()
