from ToningProcessing import getToningScore
from Brilliance import getBrilliance_And_Percent_Silver
image = 'ScrapedImages/obverse/Morgan 1879-O NGC MS62 2097792 obverse.jpg'

score = getToningScore(image)
brilliance, percent_silver = getBrilliance_And_Percent_Silver(image)
print(score)
print("Brilliance, Percent Silver: ", brilliance, percent_silver)

print('Toning Score = ', score)

