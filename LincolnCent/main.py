# Coins/main.py     Creed Jones     VT ECE  Sept 8, 2022
# development in support of the MDE coin grading teams AY22-23
# this main program is just the top level

import os
import numpy as np
import pandas as pd

from CoinImage import CoinImage
from MaskImage import MaskImage

def process_one_image(coinimg, wedges):
    M = MaskImage()
    coinfeat = np.zeros( (1, 0), dtype=float)
    for wedgect in range(len(wedges)):
        M.clear()
        M.create_wedge(start_angle=wedges[wedgect][0], end_angle=wedges[wedgect][1],
                       xc=int(coinimg.coincenter[0]), yc=int(coinimg.coincenter[1]))
        feat = np.asarray(coinimg.applymask(M)).reshape(1,-1)
        coinfeat = np.concatenate( (coinfeat, feat), axis=1)

    return coinfeat[0]

def do_main():
    dirname = 'C:/Users/Adam/Desktop/ECE 4805/Repositories/Condition_FeatureExtraction/Obverses/' #use /
    #wedges = ( (0, 90), (90, 180), (180, 270), (270, 0) ) # 4 wedges
    wedges = ( (0, 45), (45, 90), (90, 135), (135, 180), (180, 225), (225, 270), (270, 315), (315, 360) ) # 8 wedges
    I = CoinImage()
    colnames = []
    for wedgect in range(len(wedges)):
        colnames.extend(I.featurenames("W" + str(wedgect)))
    df = pd.DataFrame(columns=colnames)
    for filename in os.listdir('C:/Users/Adam/Desktop/ECE 4805/Repositories/Condition_FeatureExtraction/Obverses/'):
        print(filename)
        I.load(dirname + filename)
        I.findcenter()
        #I.save(dirname + filename.replace('.jpg', '.png'))
        df.loc[filename] = process_one_image(coinimg=I, wedges=wedges)
    df.to_excel(dirname + 'Features_Obverses_Sobel5_8Wedge.xlsx')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    do_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
