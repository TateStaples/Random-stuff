if __name__ == '__main__':
    barrier = "--------------------------------"


def begin_section(name):
    buffer = 1
    print(barrier)
    print("Section: " + name)
    for i in range(buffer):
        print()
# where to write code - go to trinket.io
# Resources
# Textbook: https://books.trinket.io/

# google - *most important lesson is that you should google everything. That is the most important skill of coding
# -------------------------------------------------------------------------------------------------------- #

# comments
  
# use "#" for a normal comment
'''
use 3 apostrophes for a multiple line comment
I can type whatever I want here
This isn't code but it is good for taking notes
adjfkafd

fdaf
dasfd
afd
stuff
'''

# -------------------------------------------------------------------------------------------------------- #
# types and variables
begin_section("variables & types")

# what a variable is, "=" explained - naming values
variable = 10
name = "Nathan"
# 10 = variable - this doesn't work because only one way

# types
number = 10  # int
print(number, type(number))  # output =

decimal = 1.4567  # float
print(decimal, type(decimal))   # output =

truth = True  # boolean - used for logic
print(truth, type(truth))   # output =

words_and_stuff = "don't have spaces in variables"  # Strings
print(words_and_stuff, type(words_and_stuff))   # output =

# -------------------------------------------------------------------------------------------------------- #
# if statements
begin_section("if statements")

if True:
    print("i did this because it is true")

if False:
    print("this wont happen because it is False")

if not True:
    print("using the word reverses it so True becomes false")

vari = 1
1 == 1
if 1 == 1:  # = and == are different
    # this happened because 1 is the same as 1
    print("1 = 1")  # output: 1 = 1
    print(1 == 1)  # output: True

# else statements
if 1 == 2:
    print("i didnt get here because 1 is not 2") # doesn't get here
else:
    print("sup nerd")  # comes here because previous not happen

# elif statements - keeps going down the list until it goes into one
if False:
    print("i didnt get here")  # doesn't get here
elif False:
    print("i didnt get here")  # doesn't get here
elif True:
    print("i got here")  # comes here
else:
    print("i didnt get here")  # doesn't get here because elif happened

# -------------------------------------------------------------------------------------------------------- #
begin_section("logic")

# and
True and True == True
if True and True:
    pass

True and False == False
if True and False:
    pass

False and False == False
if False and False:
    pass

# or
True or True == True
if True or True:
    pass

True or False == True
if True or False:
    pass

False or False == False
if False or False:
    pass

# -------------------------------------------------------------------------------------------------------- #
begin_section("loops")
# loops
x = 0
while x < 10:  # if this is true do this ->
    x = x + 1
    print(x)
    # check if should repeat
    '''
    output =
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    '''

for number in range(1, 10):  # smaller version of the while loop
    print(number)

for name in ["Tate", "Bob", "Nathan"]:
    print("Hi " + name)

# -------------------------------------------------------------------------------------------------------- #
# functions
begin_section("functions")
# basic function
def function_name():  # definition
    print("hi")  # argument
    # stuff you do
function_name()  # call


# parameter and arguments
def next(x):
    print(x+1)
next(1)  # output = 2


# return
def f(x):
    thing = x + 1
    return thing

hi = f(11)
print(hi)  # output = 12

# -------------------------------------------------------------------------------------------------------- #
begin_section("math")
# math
x = 1 + 2  # addition
y = 10 - 2  # subtraction
bob = 10 * 5  # multiplication

weird_divsion = 10 // 3  # integer division
normal_divison = 10 / 3  # normal division
print(weird_divsion)  # output = 3
print(normal_divison)  # output = 3.33333

expo = 2 ** 5  # exponents
mod = 5 % 2  # mod (remainder of division)

# -------------------------------------------------------------------------------------------------------- #

# input/output
print("output")  # output = a way to send info to user (ex. print)
your_name = input("Type something here:  ")  # allows user to enter a type String that is used during runtime
if your_name.lower() == "will":
    print("hi nerd")  # output = hi nerd if the user types Caroline


# -------------------------------------------------------------------------------------------------------- #
begin_section("weird")
# lists and more advanced types

# indexing - an index is just location of thing (first, second, etc.)
String = "12345678"
print(String.index("1")) # starts at zero, output = 0

print(String[3])  # how to get index, output = 4
print(String[1:4])  # slice (list between to indexes), output = [2, 3, 4]

# lists
my_list = [214,4123, 3214]  # can be changed
my_list.append("dkjfahd")  # how to add things
my_list.remove(214)  # how to remove things
print(my_list)  # output = [4123, 3214, "dkjfahd"]

# tuples
hi = (1, 2, 3)  # cannot be changed
a, b, c = hi  # can assign weird (not important)

# dictionaries (has a key and value)
dictionary = {"nerd": "will", "cool kid": "Tate"} # if you enter the first thing, it gives the second
print(dictionary["cool kid"])  # outputs Tate

# -------------------------------------------------------------------------------------------------------- #