"""
buildDatabase.py

~

Author: Matthew Donlon
Date: ~
Revised By: Jasper Emick
Date: 11 Mar 23
"""

import numpy as np
import os
from CoinImage import CoinImage
import ToningMacros as Macros
import cv2

colorPaletteNames = {'Light Gold': 0,
                     'Medium Gold': 0,
                     'Amber': 0,
                     'Russet': 0,
                     'Burgundy': 0,
                     'Cobalt Blue': 0,
                     'Light Cyan Blue': 0,
                     'Pale Mint Green': 0,
                     'Lemon Yellow': 0,
                     'Sunset Yellow': 0,
                     'Orange': 0,
                     'Red': 0,
                     'Magenta': 0,
                     'Magenta Blue': 0,
                     'Blue': 0,
                     'Blue Green': 0,
                     'Emerald Green': 0,
                     'Gold': 0,
                     'Medium Magenta': 0,
                     'Deep Blue': 0,
                     'Deep Green': 0,
                     'Deep Magenta': 0,
                     'Deep Purple': 0,
                     'Glossy Black': 0,
                     'Dull Black': 0,
                     'Dark Red': 0,
                     'Wine Red': 0}


def find_colors(coin):
    """
    Searches for each pixel that falls into a specified color category
    """
    image = coin.colorimg

    # Light Gold
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.LIGHT_GOLD_MIN_THRESH) &
                               np.less_equal(image, Macros.LIGHT_GOLD_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Light Gold'] = np.sum(cnt == 3)  # Checks that R, G, and B satisfy the conditions

    # Medium Gold
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.MEDIUM_GOLD_MIN_THRESH) &
                               np.less_equal(image, Macros.MEDIUM_GOLD_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Medium Gold'] = np.sum(cnt == 3)

    # Amber
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.AMBER_MIN_THRESH) &
                               np.less_equal(image, Macros.AMBER_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Amber'] = np.sum(cnt == 3)

    # Russet
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.RUSSET_MIN_THRESH) &
                               np.less_equal(image, Macros.RUSSET_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Russet'] = np.sum(cnt == 3)

    # Burgundy
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.BURGUNDY_MIN_THRESH) &
                               np.less_equal(image, Macros.BURGUNDY_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Burgundy'] = np.sum(cnt == 3)

    # Cobalt Blue
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.COBALT_BLUE_MIN_THRESH) &
                               np.less_equal(image, Macros.COBALT_BLUE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Cobalt Blue'] = np.sum(cnt == 3)

    # Light Cyan Blue
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.LIGHT_CYAN_BLUE_MIN_THRESH) &
                               np.less_equal(image, Macros.LIGHT_CYAN_BLUE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Light Cyan Blue'] = np.sum(cnt == 3)

    # Pale Mint Green
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.PALE_MINT_GREEN_MIN_THRESH) &
                               np.less_equal(image, Macros.PALE_MINT_GREEN_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Pale Mint Green'] = np.sum(cnt == 3)

    # Lemon Yellow
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.LEMON_YELLOW_MIN_THRESH) &
                               np.less_equal(image, Macros.LEMON_YELLOW_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Lemon Yellow'] = np.sum(cnt == 3)

    # Sunset Yellow
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.SUNSET_YELLOW_MIN_THRESH) &
                               np.less_equal(image, Macros.SUNSET_YELLOW_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Sunset Yellow'] = np.sum(cnt == 3)

    # Orange
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.ORANGE_MIN_THRESH) &
                               np.less_equal(image, Macros.ORANGE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Orange'] = np.sum(cnt == 3)

    # Red
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.RED_MIN_THRESH) &
                               np.less_equal(image, Macros.RED_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Red'] = np.sum(cnt == 3)

    # Magenta
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.MAGENTA_MIN_THRESH) &
                               np.less_equal(image, Macros.MAGENTA_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Magenta'] = np.sum(cnt == 3)

    # Magenta Blue
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.MAGENTA_BLUE_MIN_THRESH) &
                               np.less_equal(image, Macros.MAGENTA_BLUE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Magenta Blue'] = np.sum(cnt == 3)

    # Blue
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.BLUE_MIN_THRESH) &
                               np.less_equal(image, Macros.BLUE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Blue'] = np.sum(cnt == 3)

    # Blue Green
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.BLUE_GREEN_MIN_THRESH) &
                               np.less_equal(image, Macros.BLUE_GREEN_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Blue Green'] = np.sum(cnt == 3)

    # Emerald Green
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.EMERALD_GREEN_MIN_THRESH) &
                               np.less_equal(image, Macros.EMERALD_GREEN_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Emerald Green'] = np.sum(cnt == 3)

    # Gold
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.GOLD_MIN_THRESH) &
                               np.less_equal(image, Macros.GOLD_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Gold'] = np.sum(cnt == 3)

    # Medium Magenta
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.MEDIUM_MAGENTA_MIN_THRESH) &
                               np.less_equal(image, Macros.MEDIUM_MAGENTA_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Medium Magenta'] = np.sum(cnt == 3)

    # Deep Blue
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DEEP_BLUE_MIN_THRESH) &
                               np.less_equal(image, Macros.DEEP_BLUE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Deep Blue'] = np.sum(cnt == 3)

    # Deep Green
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DEEP_GREEN_MIN_THRESH) &
                               np.less_equal(image, Macros.DEEP_GREEN_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Deep Green'] = np.sum(cnt == 3)

    # Deep Magenta
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DEEP_MAGENTA_MIN_THRESH) &
                               np.less_equal(image, Macros.DEEP_MAGENTA_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Deep Magenta'] = np.sum(cnt == 3)

    # Deep Purple
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DEEP_PURPLE_MIN_THRESH) &
                               np.less_equal(image, Macros.DEEP_PURPLE_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Deep Purple'] = np.sum(cnt == 3)

    # Glossy Black
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.GLOSSY_BLACK_MIN_THRESH) &
                               np.less_equal(image, Macros.GLOSSY_BLACK_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Glossy Black'] = np.sum(cnt == 3)

    # Dull Black
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DULL_BLACK_MIN_THRESH) &
                               np.less_equal(image, Macros.DULL_BLACK_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Dull Black'] = np.sum(cnt == 3)

    # Dark Red
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.DARK_RED_MIN_THRESH) &
                               np.less_equal(image, Macros.DARK_RED_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Dark Red'] = np.sum(cnt == 3)

    # Wine Red
    tonedIndices = np.argwhere(np.greater_equal(image, Macros.WINE_RED_MIN_THRESH) &
                               np.less_equal(image, Macros.WINE_RED_MAX_THRESH))
    _, cnt = np.unique(tonedIndices[:, 0:2], return_counts=True, axis=0)
    colorPaletteNames['Wine Red'] = np.sum(cnt == 3)

    result = []
    print(colorPaletteNames)
    for key in colorPaletteNames.keys():
        if colorPaletteNames[key] > 0:
            result.append(key)
    print(result)
    return result


if __name__ == '__main__':
    image = os.path.abspath('Images/matt/dark-green-toned-coin-front.jpg')
    print(image)
    coin = CoinImage()
    coin.load(image)
    find_colors(coin=coin)



"""
OLD
"""

# for outerArray in image:
#     for innerArray in outerArray:
#         if 250 <= innerArray[2] <= 255 and 250 <= innerArray[1] <= 255 and 205 <= innerArray[0] <= 215:
#             colorPaletteNames['Light Gold'] += 1
#         elif 235 <= innerArray[2] <= 245 and 236 <= innerArray[1] <= 246 and 177 <= innerArray[0] <= 187:
#             colorPaletteNames['Medium Gold'] += 1
#         elif 204 <= innerArray[2] <= 214 and 171 <= innerArray[1] <= 181 and 101 <= innerArray[0] <= 111:
#             colorPaletteNames['Amber'] += 1
#         elif 123 <= innerArray[2] <= 133 and 94 <= innerArray[1] <= 104 and 8 <= innerArray[0] <= 18:
#             colorPaletteNames['Russet'] += 1
#         elif 124 <= innerArray[2] <= 134 and 59 <= innerArray[1] <= 69 and 101 <= innerArray[0] <= 112:
#             colorPaletteNames['Burgundy'] += 1
#         elif innerArray[2] <= 6 and 26 <= innerArray[1] <= 36 and 189 <= innerArray[0] <= 199:
#             colorPaletteNames['Cobalt Blue'] += 1
#         elif 187 <= innerArray[2] <= 197 and 221 <= innerArray[1] <= 231 and 250 <= innerArray[0] <= 255:
#             colorPaletteNames['Light Cyan Blue'] += 1
#         elif 220 <= innerArray[2] <= 230 and 250 <= innerArray[1] <= 255 and 219 <= innerArray[0] <= 229:
#             colorPaletteNames['Pale Mint Green'] += 1
#         elif 245 <= innerArray[2] <= 255 and 245 <= innerArray[1] <= 255 and 186 <= innerArray[0] <= 226:
#             colorPaletteNames['Lemon Yellow'] += 1
#         elif 244 <= innerArray[2] <= 255 and 215 <= innerArray[1] <= 235 and 13 <= innerArray[0] <= 33:
#             colorPaletteNames['Sunset Yellow'] += 1
#         elif 250 <= innerArray[2] <= 255 and 186 <= innerArray[1] <= 196 and 18 <= innerArray[0] <= 28:
#             colorPaletteNames['Orange'] += 1
#         elif 240 <= innerArray[2] <= 255 and innerArray[1] <= 15 and innerArray[0] <= 15:
#             colorPaletteNames['Red'] += 1
#         elif 219 <= innerArray[2] <= 229 and innerArray[1] <= 5 and 123 <= innerArray[0] <= 133:
#             colorPaletteNames['Magenta'] += 1
#         elif 188 <= innerArray[2] <= 198 and 59 <= innerArray[1] <= 69 and 250 <= innerArray[0] <= 255:
#             colorPaletteNames['Magenta Blue'] += 1
#         elif innerArray[2] <= 5 and 93 <= innerArray[1] <= 103 and 219 <= innerArray[0] <= 229:
#             colorPaletteNames['Blue'] += 1
#         elif innerArray[2] <= 10 and 155 <= innerArray[1] <= 165 and 123 <= innerArray[0] <= 133:
#             colorPaletteNames['Blue Green'] += 1
#         elif innerArray[2] <= 6 and 189 <= innerArray[1] <= 199 and innerArray[0] <= 6:
#             colorPaletteNames['Emerald Green'] += 1
#         elif 220 <= innerArray[2] <= 230 and 219 <= innerArray[1] <= 229 and innerArray[0] <= 6:
#             colorPaletteNames['Gold'] += 1
#         elif 188 <= innerArray[2] <= 198 and innerArray[1] <= 5 and 120 <= innerArray[0] <= 130:
#             colorPaletteNames['Medium Magenta'] += 1
#         elif innerArray[2] <= 7 and innerArray[1] <= 5 and 218 <= innerArray[0] <= 228:
#             colorPaletteNames['Deep Blue'] += 1
#         elif innerArray[2] <= 6 and 123 <= innerArray[1] <= 133 and innerArray[0] <= 6:
#             colorPaletteNames['Deep Green'] += 1
#         elif 125 <= innerArray[2] <= 135 and innerArray[1] <= 5 and 123 <= innerArray[0] <= 133:
#             colorPaletteNames['Deep Magenta'] += 1
#         elif 58 <= innerArray[2] <= 68 and innerArray[1] <= 6 and 123 <= innerArray[0] <= 133:
#             colorPaletteNames['Deep Purple'] += 1
#         elif innerArray[2] <= 5 and innerArray[1] <= 5 and 58 <= innerArray[0] <= 68:
#             colorPaletteNames['Glossy Black'] += 1
#         elif innerArray[2] <= 5 and innerArray[1] <= 5 and innerArray[0] <= 5:
#             colorPaletteNames['Dull Black'] += 1
#         elif innerArray[2] <= 5 and innerArray[1] <= 5 and 135 <= innerArray[0] <= 145:
#             colorPaletteNames['Dark Red'] += 1
#         elif 11 <= innerArray[2] <= 21 and 20 <= innerArray[1] <= 30 and 79 <= innerArray[0] <= 89:
#             colorPaletteNames['Wine Red'] += 1