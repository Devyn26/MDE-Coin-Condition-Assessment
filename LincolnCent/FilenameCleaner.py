"""
MDE 2022-2023 Indian Head Cent Classifier
Filename Cleaner

Adjusts filenames from Scraper outputs to match manually scraped nomenclature
    <InventoryNumber Grader YearMintmark Grade Color EyeAppeal Toning PictureType (Special Notes)>.jpg
    Ex. "2348850 PCGS 1860 MS64+ CN 5 2 Reverse (Pointed Bust).jpg"

TO USE:
    Place in EMPTY directory along with .jpg images scraped from DLRC and run
    
    Only for use in tandem with IHC Scraper as of 10/18/2022
    Currently supports Obverse and Reverse images only
    Does not currently support Details coins
    Retain original scraped images for verification in a separate archival directory

Adam Pratte
pratteadam23@vt.edu    
"""

import os

directory = os.getcwd()

for filename in os.listdir(directory):
    if (filename != 'FilenameCleaner.py'):                              # prevents the script file itself from being modified
        substrings = filename.split()                                   # break each filename into a list of strings for reordering/removal
        #print(substrings)
        
        # -------------- Inventory Number -----------

        invNum = substrings[0]
        if(invNum[0].isalpha() and invNum[1].isalpha()):                # shave the color identifier from the inventory number if still present
            invNum = invNum[2:] + " "
        #print(invNum)
        
        # -------------- Date/Mintmark --------------

        date = substrings[1] + " "
        if (date == "1909-S "):
            date = "1909S "
        elif (date == "1908-S "):
            date = "1908S "
        elif (date == "1864-L"):
            date = "1864L "
        #print(date)

        if (date == "1909 " or date == "1909S "):
            substrings.remove("Indian")                                 # DLRC includes this to differentiate between 1909 Lincoln and IH Cents
        substrings.remove("1c")                                         # not useful for data analysis, we know it's a penny
        #print(substrings)

        # ------------ Grading Company --------------

        grader = substrings[2] + " "                                    # no modifications necessary to grader

        # ----------- Sheldon Scale Grade -----------

        grade = substrings[3]

        if "-" in grade:                                                # standardize grades to Sheldon Scale grade codes (PO, FR, G,..., MS)
            grade = grade.replace("-", "")
        if "Good" in grade:
            grade = grade.replace("Good", "G")
        if "Fair" in grade:
            grade = grade.replace("Fair", "FR")
        if "Poor" in grade:
            grade = grade.replace("Poor", "PO")  
        
        grade = grade + " "
        #print(grade)    

        # ---------------- Color -------------------

        notesOffset = 0                                                 # prevents out of bounds memory access, later substrings can be from different categories

        if substrings[4] == "BN" or substrings[4] == "RB" or substrings[4] == "RD":
            color = substrings[4]
            notesOffset = notesOffset + 1                               # with color present, check [5] for parentheses
        else:
            if (date == "1859 " or date == "1860 " or date == "1861 " or date == "1862 " or date == "1863 " or date == "1864 "):
                color = "CN"                                            # no offset increase when color is missing
            else:
                color = "BN"                                
        color = color + " "
        #print(color)

        # ------------ Auctioneer Notes ------------

        notes = ""                                                      # by default empty, is appended to if "( )" detected

        if "(" in substrings[4 + notesOffset]:
            notes = " "
            while (")" not in substrings[4 + notesOffset]):             # while loop to concatenate notes, increment offset
                notes = notes + substrings[4 + notesOffset] + " "       # USER NOTE: This assumes that the Scraper has provided ( and ) in the filename strings
                notesOffset = notesOffset + 1

            notes = notes + substrings[4 + notesOffset]                 # append current held value and add a spacer
            notesOffset = notesOffset + 1                               # additional offset for held value
        else:                                                           # no parentheses found, skip extraneous data until numeric Eye Appeal Score is located
            while not(substrings[4 + notesOffset].isnumeric()):
                notesOffset = notesOffset + 1

        # -------------- Eye Appeal --------------

        eyeAppeal = substrings[4 + notesOffset] + " "                   # still at 4 because of the extra offset 

        # ---------------- Toning ----------------

        toning = substrings[5 + notesOffset] + " "

        # -------------- PictureType --------------

        pictureType = substrings[6 + notesOffset]
        pictureType = pictureType[:-4]                                  # remove ".jpg", goes at the end

        #print(invNum + grader + date + grade + color + eyeAppeal + toning + pictureType + notes + ".jpg")
        newFilename = invNum + grader + date + grade + color + eyeAppeal + toning + pictureType + notes + ".jpg"
        
        os.rename(filename, newFilename)                                # my favorite line