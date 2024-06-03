#bug 0, cause an exception to when run
def fReplaceText(strInput, strText, intStart, intStop):
    """
    Replace the text between intStart and intStop with strText
    """
    #copy the string
    strReturn = strInput[:]
    #update the section of the string with inputted text
    strReturn[intStart:intStop] = strText
    return strReturn

def fFindAndReplaceInStr(strInput, strOld, strNew):
    """
    replace strOld with strNew in strInput
    """
    intLenOld = len(strOld)
    intStart = strInput.find(strOld)
    fReplaceText(strInput, strNew, intStart, intStart+intLenOld)


#fix famous misquote
print(fFindAndReplaceInStr("Luke, I am your father.", 'Luke', 'No'))
#Hint: run the file in debug mode, VSCode should catch the exception.
#Then try the Error line of code in DEBUG CONSOLE, check these strings, how to make it right?
