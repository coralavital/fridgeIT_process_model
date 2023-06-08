import datetime
import re

def  run_expiration_date_workflow(string):
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
        if result != None:
            date = fix_date(re.sub(r"\D", "", result[0]))

    elif result != None:
        date = re.sub(r"[^\d]", "/", result[0])
        if date.count('/') < 2 :
            date = fix_date(re.sub(r"\D", "", result[0]))


    if date.count('/') == 2:
        year = '20'+date[-2:]
        date = date[:-2] + year


    elif date.count('/') == 1:
        date += '/' + datetime.date.today().strftime("%Y")

    if date == '':
        return result[0] if result != None else 'not founded'
        
    return date if result != None else 'not founded'



def fix_date(date:str):
    """
    changing the format of the date to be dd/mm \n
    if the numbers of the date are non sense for example month or day with hte number 77\n
    it will return not founded\n
    Args:   
        date (str): the numbers of the date only

    Returns:
        str: dd/mm date format or not founded
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
        return "not founded"
        
    return f"{day}/{month}"
