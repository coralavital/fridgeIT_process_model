import datetime
import re


ERROR_MSG = "not found"

def split_string(input_string):
    # pattern = r"[,./-:d]+"  # Regular expression pattern to match numbers, hyphens, forward slashes, and periods
    pattern = r"\b(\d{0,4}[-,.:/]\d{0,4}(?:[-,.:/]\d{0,4})?)\b"
    split_values = re.findall(pattern, input_string)
    pattern = r"\d+"  # Regular expression pattern to match one or more digits
    result = [string for string in split_values if re.search(pattern, string) and len(string) > 4]
    return result

def is_valid_date_format(str):
    date_without_year = r"^\d{0,2}[-,.:/]\d{0,2}$"
    date_with_year = r"^\d{0,2}[-,.:/]\d{0,2}[-,.:/]\d{0,4}$"
    flag1 = re.match(date_without_year, str)
    flag2 = re.match(date_with_year, str)
    if bool(flag1) or bool(flag2):
        return True
    else:
        return False

def is_valid_date(date_string):
    try:
        datetime.datetime.strptime(date_string, "%d/%m/%Y")
        return True
    except ValueError:
        return False

def  run_expiration_date_workflow(string):
    """
        find the date in format dd/mm or dd/mm/yyyy from the given string using regex
    Args:
        string (str): string with date

    Returns:
        str: the date that found from the string in format dd/mm/yyyy or not found
    """
    # Define regex pattern for date format
    print("string ===== ",string )
    strings = split_string(string)
    valid_dates = []


    # all the dates will have / as split
    for char in [',','.','/','-',':']:
        for i in range(len(strings)):
            if len(strings[i]) > 2:
                strings[i] = strings[i].replace(char, "/")


    for i in range(len(strings)):
        if not is_valid_date_format(str=strings[i]):
            if fix_date(strings[i][:4]) == ERROR_MSG:
                strings[i] = ERROR_MSG
            else: 
                strings[i] = fix_date(strings[i][:4]) + strings[i][4:]


    # all the dates will include year
    for i in range(len(strings)):
        # numbers -> date , fix date func
        if ERROR_MSG in strings[i]:
            continue
        if strings[i].count('/') == 0:
            strings[i] = fix_date(strings[i])
            if strings[i] != ERROR_MSG:
                strings[i] += '/' + datetime.date.today().strftime("%Y")
                valid_dates.append(strings[i])

        elif strings[i].count('/') == 1:
            strings[i] += '/' + datetime.date.today().strftime("%Y")
            valid_dates.append(strings[i])

        elif strings[i].count('/') == 2:
            year = '20'+strings[i][-2:]
            valid_dates.append(strings[i][:-2] + year)


    result=[]
    result = [valid_date for valid_date in valid_dates if is_valid_date(valid_date)]
    print("##########")
    if result:
        print(result[0])
        print("##########")
        return result[0]
    else: 
        print(ERROR_MSG)
        print("##########")
        return ERROR_MSG
    


def fix_date(date:str):
    """
    changing the format of the date to be dd/mm \n
    if the numbers of the date are non sense for example month or day with hte number 77\n
    it will return not found\n
    Args:   
        date (str): the numbers of the date only

    Returns:
        str: dd/mm date format or not found
    """
    try:
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
            return ERROR_MSG
            
        return f"{day}/{month}"
    except:
        return ERROR_MSG