#Bug2
#It's a silent bug (no exception)
#Please modify build_new_dictionary_from_input function to correct it
def build_new_dictionary_from_input(dIn):
    """Add keyword Total to a dictionary, return an updated dictionary
    Don't change the input

    Args:
        dIn (dictionray): 

    Returns:
        updated dictionary
    """
    nTotal = 0
    for key in dIn.keys():
        nTotal += dIn[key]
    #Adding total to the dictionary
    dIn['Total'] = nTotal
    return dIn

dOuput = {}
dInput = {'A':1,
        'B':2,
        'C':3}
dOutput = build_new_dictionary_from_input(dInput)
print('Input dictionary: ', dInput)
print('Onput dictionary: ', dOutput)

if dInput == {'A':1, 'B':2, 'C':3} and dOutput == {'A':1, 'B':2, 'C':3, 'Total':6}:
    print('Test passed')
else:
    print('Test fail')
if 'Total' in dInput.keys():
    print('Error, input should not change')
#Do you see something strange here?
#Hint: add dInput, dOuput to WATCH, place breakpoint at line #14, see what changes after 1 and 2 step over
#Use ID() command to check if the 2 objects dInput and dOutput are the same