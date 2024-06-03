#Bug1
#Running the file leads to an Exception
#Please modify the del_odd_number function to fix the bug
def del_odd_number(list_of_numbers):
    """Remove all odd numbers from a list_of_numbers

    Args:
        list_of_numbers (list): input list

    Returns:
        updated list with no odd number
    """
    for idx in range(len(list_of_numbers)):
        if list_of_numbers[idx]%2:
            del list_of_numbers[idx]
    return list_of_numbers

list_of_numbers = list(range(10))
print('Input list:  ', list_of_numbers)
output_list = del_odd_number(list_of_numbers)

#Hint: place a break point at the following line of code
#Add list_of_nunber to the Watch list
#Step into the function del_odd_number, repeating step over, pay attention to the index
print('Output list: ', del_odd_number(list_of_numbers))
if output_list == [0,2,4,6,8]:
    print('Test passed')
else:
    print('Test failed, correct output should be ', [0,2,4,6,8])