from transformers import pipeline
import re

captioner = pipeline("image-to-text",model="jinhybr/OCR-Donut-CORD")


def fix_date(date:str):
    """
    changing the format of the date to be dd/mm\n
    if the numbers of the date are non sense for example month or day with hte number 77\n
    it will return not found\n
    Args:   
        date (str): the numbers of the date only

    Returns:
        str: dd/mm date format or not found
    """
    day = ''
    month = ''
    
    if date[0] == '0':
        day = date[1]
        if date[2] == '0':
            month = date[3]
        else:
            month = date[2:4]
    else:
        day = date[:2]
        if date[2] == '0':
            month = date[3]
        else:
            month = date[2:4]

    #   in case we got invalid date
    if int(month) > 12 or int(day) > 31:
        return "not found"
        
    return f"{day}/{month}"


def  is_date(string:str):
    """
        find the date in format dd/mm or dd/mm/yyyy from the given string using regex
    Args:
        string (str): string with date

    Returns:
        str: the date that found from the string or not found
    """
    # Define regex pattern for date format
    date_pattern_with_year = r'\d{1,2}[,./-:]\d{1,2}[,./-:]\d{1,4}'
    result = re.search(date_pattern_with_year, string)
    date = ''

    if result == None:
        date_pattern = r'\d{1,2}[,./-:]\d{1,2}([,./-:]\d{1,4})?'
        result = re.search(date_pattern, string)
    elif result != None:
        date = re.sub(r"[^\d]", "/", result[0])
        if date.count('/') < 2 :
            date = fix_date(re.sub(r"\D", "", result[0]))
    if date == '':
        return result[0] if result != None else 'not found'
        
    return date if result != None else 'not found'


from PIL import Image

import os

# Get the current directory
dir_path = os.getcwd()

# List all files in the directory
all_files = os.listdir(dir_path)

# Filter out only the PNG files
png_files = [file for file in all_files if file.endswith('.png')]
print(png_files)
for png_img in png_files:
    # Load the image
    image = Image.open(png_img)

    data = captioner(f"{png_img}")

    # print(f"{png_img} string sent:", data[0]['generated_text'])
    print(f"{png_img} = ", is_date(data[0]['generated_text']))