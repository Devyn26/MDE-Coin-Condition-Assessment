'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 3/05/2025
'''
from fpdf import FPDF
import cv2
import os
from PIL import Image
import tempfile
# A4 orientation Standard
PDF_WIDTH = 210
PDF_HEIGHT = 297
STUPID_INDENT = "\t\t\t\t\t\t\t"


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.ogObverse = None
        self.ogReverse = None
        self.conditionObverse = None
        self.conditionReverse = None
        self.flatObverse = None
        self.flatReverse = None
        self.condMasks = {"highSigObverse": None, "highSigReverse": None, "lowSigObverse": None, "lowSigReverse": None,
                          "rimObverse": None, "rimReverse": None}
        self.toningObverse = None
        self.toningReverse = None
        self.obToningCovImgDR = None
        self.reToningCovImgDR = None
        self.brillianceObverse = None
        self.brillianceReverse = None

        self.conditionScore = None
        self.brillianceScore = None
        self.toningScore = None
        self.histBrillliance = None
    def genTextPageOne(self):
        # Title: "Detailed Results"
        self.set_xy(0.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='C', txt="Detailed Results", border=0)

        # Sub-Title: "Condition Score"
        self.set_xy(20.0, 105.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Condition Analysis: ", border=0)

        # TODO: Get coin grade and add it to a cell
        coin_grade = "Condition Score: " + str(self.conditionScore) + "/70.0"
        self.set_xy(20.0, 130.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=coin_grade, border=0)

        # Description: Condition
        desc = STUPID_INDENT + "The condition of a coin is evaluated by the frequency of white pixels (Intensity: " \
                               "255)\nfound within an edge filtered image. This evaluation is applied multiple " \
                               "times with different\nspecialized coin masks."
        self.set_xy(20.0, 140.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=desc, border=0)

    def genImagesPageOne(self):
        # Original Input coin Images
        self.rect(20.0, 35.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 35.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_obverse:
            self.ogObverse.save(tmp_file_obverse.name, 'PNG')  # Save as PNG (or any other format)
            tmp_file_obverse.close()  # Close the file so it can be accessed by FPDF
            self.image(tmp_file_obverse.name, w=80, h=80)

        self.rect(110.0, 35.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 35.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_reverse:
            self.ogReverse.save(tmp_file_reverse.name, 'PNG')
            tmp_file_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_reverse.name, w=80, h=80)

        # Raw edge filter (no mask)
        self.rect(20.0, 155.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 155.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_condition_obverse:
            condition_obverse_image = Image.fromarray(self.conditionObverse)
            condition_obverse_image.save(tmp_file_condition_obverse.name, 'PNG')
            tmp_file_condition_obverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_condition_obverse.name, w=80, h=80)

        self.rect(110.0, 155.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 155.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_condition_reverse:
            condition_reverse_image = Image.fromarray(self.conditionReverse)
            condition_reverse_image.save(tmp_file_condition_reverse.name, 'PNG')
            tmp_file_condition_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_condition_reverse.name, w=80, h=80)

    def genTextPageTwo(self):
        # Sub-Title: "Flat Regions"
        self.set_xy(20.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Flat Regions Mask", border=0)

        # Sub-Title: "High Significance Details"
        self.set_xy(20.0, 110.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="High Significance Detail Mask", border=0)

    def genImagesPageTwo(self):
        # Flat mask
        self.rect(20.0, 40.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 40.0)
        # Save flatObverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_flat_obverse:
            flat_obverse_image = Image.fromarray(self.flatObverse)
            flat_obverse_image.save(tmp_file_flat_obverse.name, 'PNG')
            tmp_file_flat_obverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_flat_obverse.name, w=80, h=80)

        self.rect(110.0, 40.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 40.0)
        # Save flatReverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_flat_reverse:
            flat_reverse_image = Image.fromarray(self.flatReverse)
            flat_reverse_image.save(tmp_file_flat_reverse.name, 'PNG')
            tmp_file_flat_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_flat_reverse.name, w=80, h=80)

        # Red/Orange mask
        self.rect(20.0, 150.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 150.0)
        # Save highSigObverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_high_sig_obverse:
            high_sig_obverse_image = Image.fromarray(self.condMasks["highSigObverse"])
            high_sig_obverse_image.save(tmp_file_high_sig_obverse.name, 'PNG')
            tmp_file_high_sig_obverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_high_sig_obverse.name, w=80, h=80)

        self.rect(110.0, 150.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 150.0)
        # Save highSigReverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_high_sig_reverse:
            high_sig_reverse_image = Image.fromarray(self.condMasks["highSigReverse"])
            high_sig_reverse_image.save(tmp_file_high_sig_reverse.name, 'PNG')
            tmp_file_high_sig_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_high_sig_reverse.name, w=80, h=80)

    def genTextPageThree(self):
        # Sub-Title: "Low Significance Details"
        self.set_xy(20.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Low Significance Detail Mask", border=0)

        # Sub-Title: "Rim Details"
        self.set_xy(20.0, 110.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Coin Rim Mask", border=0)

    def genImagesPageThree(self):
        # Yellow mask
        self.rect(20.0, 40.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 40.0)
        # Save lowSigObverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_low_sig_obverse:
            low_sig_obverse_image = Image.fromarray(self.condMasks["lowSigObverse"])
            low_sig_obverse_image.save(tmp_file_low_sig_obverse.name, 'PNG')
            tmp_file_low_sig_obverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_low_sig_obverse.name, w=80, h=80)

        self.rect(110.0, 40.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 40.0)
        # Save lowSigReverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_low_sig_reverse:
            low_sig_reverse_image = Image.fromarray(self.condMasks["lowSigReverse"])
            low_sig_reverse_image.save(tmp_file_low_sig_reverse.name, 'PNG')
            tmp_file_low_sig_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_low_sig_reverse.name, w=80, h=80)

        # Green mask
        self.rect(20.0, 150.0, 80.0, 80.0, 'D')
        self.set_xy(20.0, 150.0)
        # Save rimObverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_rim_obverse:
            rim_obverse_image = Image.fromarray(self.condMasks["rimObverse"])
            rim_obverse_image.save(tmp_file_rim_obverse.name, 'PNG')
            tmp_file_rim_obverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_rim_obverse.name, w=80, h=80)

        self.rect(110.0, 150.0, 80.0, 80.0, 'D')
        self.set_xy(110.0, 150.0)
        # Save rimReverse image to a temporary file and use it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_rim_reverse:
            rim_reverse_image = Image.fromarray(self.condMasks["rimReverse"])
            rim_reverse_image.save(tmp_file_rim_reverse.name, 'PNG')
            tmp_file_rim_reverse.close()  # Close the file for FPDF to access
            self.image(tmp_file_rim_reverse.name, w=80, h=80)

    def genTextPageFour(self):
        # Sub-Title: "Brightness"
        self.set_xy(20.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Brightness: ", border=0)

        # TODO: Get brightness score and add it to a cell (right now it says none but it is in the right position)
        coin_grade = "Brightness Score: " + str(self.brillianceScore) + "/10.0"
        self.set_xy(20.0, 25.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=coin_grade, border=0)

        # Description: Brightness
        desc = STUPID_INDENT + "The brightness is determined by looking at the Value from the HSV reading of the\n" \
                               "coin. There are more scores of 10 due to grading on the higher end of the Sheldon\n"\
                               "Scale, so our database incorporates many pristine coins. Below is a histogram of\n"\
                               "the brightness distribution of every coin in the database. "
        self.set_xy(20.0, 35.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=desc, border=0)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_hist_brilliance:
            hist_brilliance_image = Image.fromarray(self.histBrillliance)
            hist_brilliance_image.save(tmp_file_hist_brilliance.name, 'PNG')
            tmp_file_hist_brilliance.close()  # Close the file for FPDF to access
            self.set_xy(55.0, 55.0)
            self.image(tmp_file_hist_brilliance.name, w=110, h=70)

        # Sub-Title: "Toning Coverage"
        self.set_xy(20.0, 110.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Toning Coverage: ", border=0)

        # TODO: Get toning coverage and add it to a cell (right now it says none but it is in the right position)
        coin_grade = "Toning Score: " + str(self.toningScore) + "/10.0"
        self.set_xy(20.0, 135.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=coin_grade, border=0)

        # Description: Toning Coverage
        desc = STUPID_INDENT + "Toning coverage is calculated by looking at every pixel and running our algorithm " \
                               "that\nchecks saturation and hue values to determine whether the pixel is toned or " \
                               "not. We\ncalculate the percentage covered along with the amount of colors in the coin. "\
                               "If there are \nnot many colors but it is decently toned that will have a major effect on the score \noutput. "\
                               "The score will be the average of the obverse and reverse scores"
        self.set_xy(20.0, 145.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=desc, border=0)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_ob_toning_cov:
            ob_toning_cov_image = Image.fromarray(self.obToningCovImgDR)
            ob_toning_cov_image.save(tmp_file_ob_toning_cov.name, 'PNG')
            tmp_file_ob_toning_cov.close()  # Close the file for FPDF to access
            self.set_xy(20.0, 180.0)
            self.image(tmp_file_ob_toning_cov.name, w=80, h=80)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file_re_toning_cov:
            re_toning_cov_image = Image.fromarray(self.reToningCovImgDR)
            re_toning_cov_image.save(tmp_file_re_toning_cov.name, 'PNG')
            tmp_file_re_toning_cov.close()  # Close the file for FPDF to access
            self.set_xy(110.0, 180.0)
            self.image(tmp_file_re_toning_cov.name, w=80, h=80)

    def genImagesPageFour(self):
        # Brightness Histogram
        # TODO: Add histogram image or import from module which develops it
        # Toning Coverage Obverse
        self.rect(20.0, 180.0, 80.0, 80.0, 'D')
        self.rect(110.0, 180.0, 80.0, 80.0, 'D')

    def genTextPageFive(self):
        # Sub-Title: "Colors"
        self.set_xy(20.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=40.0, align='L', txt="Colors: ", border=0)

        # TODO: Get color list and add it to a cell

        # Description: colors
        desc = STUPID_INDENT + "Below is an image of our color palette with 25 toned colors the coin could contain. " \
                               "If\nthe coin is less than 10% toned the only color output will be silver. "
        self.set_xy(20.0, 25.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='L', txt=desc, border=0)

    #def genImagesPageFive(self):
        # Toning Coverage Reverse
        #self.rect(20.0, 20.0, 80.0, 80.0, 'D')
        #self.rect(110.0, 20.0, 80.0, 80.0, 'D')
        # TODO: Add color palette image


def generateTemplate(pdf):
    # Page 1
    pdf.add_page()
    pdf.genTextPageOne()
    pdf.genImagesPageOne()
    # Page 2
    pdf.add_page()
    pdf.genTextPageTwo()
    pdf.genImagesPageTwo()
    # Page 3
    pdf.add_page()
    pdf.genTextPageThree()
    pdf.genImagesPageThree()
    # Page 4
    pdf.add_page()
    pdf.genTextPageFour()
    pdf.genImagesPageFour()
    # Page 5
    pdf.add_page()
    pdf.genTextPageFive()
    #pdf.genImagesPageFive()

    pdf.output('MorganSilverDollar/Morgan_Dollar_main/test.pdf')


if __name__ == "__main__":
    testPDF = PDF()

    oPath = os.path.abspath('ScrapedImages/obverse') + '\\'
    oImg = cv2.imread(oPath + "Morgan 1881-S NGC MS65 2363354 obverse.jpg")
    oImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2RGB)
    testPDF.ogObverse = Image.fromarray(oImg)

    rPath = os.path.abspath('ScrapedImages/reverse') + '\\'
    rImg = cv2.imread(rPath + "Morgan 1881-S NGC MS65 2363354 reverse.jpg")
    rImg = cv2.cvtColor(rImg, cv2.COLOR_BGR2RGB)
    testPDF.ogReverse = Image.fromarray(rImg)

    generateTemplate(testPDF)
