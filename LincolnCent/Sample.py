from ToningHSV import getToningFromImg
from colorSubsystem import oneCoinRed

if __name__ == '__main__':
    filepath = 'C:/Users/jdtyl/OneDrive/Documents/2022Fall/MDE/Git-Repo-Indian-Head/Dataset/Obverse/'
    extention = '.jpg'
    # filenames = ['2287806 NGC 1904 MS62 BN 5 4 Obverse','2423860 NGC 1894 MS64 RB 4 3 Obverse','2419331 PCGS 1901 MS64 RD 5 2 Obverse (OGH)']
    filenames = ['2279073 NGC 1885 MS66 BN 5 4 Obverse','2423860 NGC 1894 MS64 RB 4 3 Obverse','2419331 PCGS 1901 MS64 RD 5 2 Obverse (OGH)']
    for filename in filenames:
        file = filepath+filename+extention
        Toning = getToningFromImg(file)
        Color = oneCoinRed(file)
        print(filename)
        print("Toning:",Toning)
        print("Color:",Color,"% Red")