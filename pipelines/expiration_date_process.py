import re

def  run_expiration_date_workflow(string, class_name):
    """
        find the date in format dd/mm from the given string
    Args:
        string (str): string with date

    Returns:
        str: the date that found from the string or not found
    """
    print(class_name)
    if class_name == 'milk':
        date_pattern = r'\d{0,9}/\d{0,9}'
        result = re.search(date_pattern, string)
    elif class_name == 'butter':
        # Define regex pattern for date format
        date_pattern = r'\d{1,2}[.]\d{1,2}[.]\d{1,2}'
        result = re.search(date_pattern, string)
    elif class_name == 'cottage':
        date_pattern = r'\d{1,2}[/]\d{1,2}'
        result = re.search(date_pattern, string)
    elif class_name == 'mustard':
        date_pattern = r'\d{1,2}[/]\d{1,2}([/]\d{1,4})?'
        result = re.search(date_pattern, string)
    elif class_name == 'cream':
        date_pattern = r'\d{1,2}[./]\d{1,2}([./-]\d{1,2})?'
        result = re.search(date_pattern, string)

    return result[0] if result != None else 'not found'