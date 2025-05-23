import numpy as np

# [B, G, R]
# Notes Lemon Yellow completely encompasses light gold in range making it basically irrelevant
LIGHT_GOLD_MIN_THRESH = np.array([205, 250, 250])
LIGHT_GOLD_MAX_THRESH = np.array([215, 255, 255])

MEDIUM_GOLD_MIN_THRESH = np.array([177, 236, 235])
MEDIUM_GOLD_MAX_THRESH = np.array([187, 246, 245])

AMBER_MIN_THRESH = np.array([101, 171, 204])
AMBER_MAX_THRESH = np.array([111, 181, 214])

RUSSET_MIN_THRESH = np.array([8, 94, 123])
RUSSET_MAX_THRESH = np.array([18, 104, 133])

BURGUNDY_MIN_THRESH = np.array([101, 59, 124])
BURGUNDY_MAX_THRESH = np.array([112, 69, 134])

COBALT_BLUE_MIN_THRESH = np.array([189, 26, 0])
COBALT_BLUE_MAX_THRESH = np.array([199, 36, 6])

LIGHT_CYAN_BLUE_MIN_THRESH = np.array([250, 221, 187])
LIGHT_CYAN_BLUE_MAX_THRESH = np.array([255, 231, 197])

PALE_MINT_GREEN_MIN_THRESH = np.array([219, 250, 220])
PALE_MINT_GREEN_MAX_THRESH = np.array([229, 255, 230])

LEMON_YELLOW_MIN_THRESH = np.array([186, 245, 245])
LEMON_YELLOW_MAX_THRESH = np.array([226, 255, 255])

SUNSET_YELLOW_MIN_THRESH = np.array([13, 215, 244])
SUNSET_YELLOW_MAX_THRESH = np.array([33, 235, 255])

ORANGE_MIN_THRESH = np.array([18, 186, 250])
ORANGE_MAX_THRESH = np.array([28, 196, 255])

RED_MIN_THRESH = np.array([0, 0, 240])
RED_MAX_THRESH = np.array([15, 15, 255])

MAGENTA_MIN_THRESH = np.array([123, 0, 219])
MAGENTA_MAX_THRESH = np.array([133, 5, 229])

MAGENTA_BLUE_MIN_THRESH = np.array([250, 59, 188])
MAGENTA_BLUE_MAX_THRESH = np.array([255, 69, 198])

BLUE_MIN_THRESH = np.array([219, 93, 0])
BLUE_MAX_THRESH = np.array([229, 103, 5])

BLUE_GREEN_MIN_THRESH = np.array([123, 155, 0])
BLUE_GREEN_MAX_THRESH = np.array([133, 165, 10])

EMERALD_GREEN_MIN_THRESH = np.array([0, 189, 0])
EMERALD_GREEN_MAX_THRESH = np.array([6, 199, 6])

GOLD_MIN_THRESH = np.array([0, 219, 220])
GOLD_MAX_THRESH = np.array([6, 229, 230])

MEDIUM_MAGENTA_MIN_THRESH = np.array([120, 0, 188])
MEDIUM_MAGENTA_MAX_THRESH = np.array([130, 5, 198])

DEEP_BLUE_MIN_THRESH = np.array([218, 0, 0])
DEEP_BLUE_MAX_THRESH = np.array([228, 5, 7])

DEEP_GREEN_MIN_THRESH = np.array([0, 123, 0])
DEEP_GREEN_MAX_THRESH = np.array([6, 133, 6])

DEEP_MAGENTA_MIN_THRESH = np.array([123, 0, 125])
DEEP_MAGENTA_MAX_THRESH = np.array([133, 5, 135])

DEEP_PURPLE_MIN_THRESH = np.array([123, 0, 58])
DEEP_PURPLE_MAX_THRESH = np.array([133, 6, 68])

GLOSSY_BLACK_MIN_THRESH = np.array([58, 0, 0])
GLOSSY_BLACK_MAX_THRESH = np.array([68, 5, 5])

DULL_BLACK_MIN_THRESH = np.array([0, 0, 0])
DULL_BLACK_MAX_THRESH = np.array([5, 5, 5])

DARK_RED_MIN_THRESH = np.array([135, 0, 0])
DARK_RED_MAX_THRESH = np.array([145, 0, 0])

WINE_RED_MIN_THRESH = np.array([79, 20, 11])
WINE_RED_MAX_THRESH = np.array([89, 30, 21])

# End Color palette Macros

HUE_THRESH_1 = 0
HUE_THRESH_2 = 10
HUE_THRESH_2P1 = 40
HUE_THRESH_2P2 = 50
HUE_THRESH_2P3 = 60
HUE_THRESH_3 = 70
HUE_THRESH_4 = 340

SAT_THRESH_1 = 10
SAT_THRESH_2 = 16
SAT_THRESH_2P1 = 35
SAT_THRESH_2P2 = 45
SAT_THRESH_2P3 = 35
SAT_THRESH_3 = 15

VALUE_THRESH = 10

# End Coverage Macros
