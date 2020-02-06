# comments
  
# use "#" for a normal comment
'''
use 3 apostrophes for a multiple line comment
I can type whatever I want here
blag
fadsfa
d
adf
'''

# -------------------------------------------------------------------------------------------------------- #
# types and variables

# what a variable is, "=" explained
variable = 10
# 10 = variable - this doesn't work

# types
number = 10  # int
print(number, type(number))  # output =

decimal = 1.4567  # float
print(decimal, type(decimal))   # output =

truth = True  # boolean
print(truth, type(truth))   # output =

words_and_stuff = "don't have spaces in variables"  # Strings
print(words_and_stuff, type(words_and_stuff))   # output =

# -------------------------------------------------------------------------------------------------------- #

# if statements

if 1 == 1:  # = and == are different
    print("1 = 1")  # output: 1 = 1
    print(1 == 1)  # output: True

# else statements
if 1 == 2:
    print("i didnt get here") # doesn't get here
else:
    print("sup nerd")  # comes here because previous not happen

# elif statements
if False:
    print("i didnt get here")  # doesn't get here
elif False:
    print("i didnt get here")  # doesn't get here
elif True:
    print("i got here")  # comes here
else:
    print("i didnt get here")  # doesn't get here because elif happened

# -------------------------------------------------------------------------------------------------------- #

# loops
x = 0
while x < 10:
    x = x + 1
    print(x)
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

set_of_vals = [1, "nerd", True, 21345.534]
for thing in set_of_vals:
    print(thing)
    '''
    output =
    1
    nerd
    True
    21345.534
    '''

# -------------------------------------------------------------------------------------------------------- #
# functions

# basic function
def function_name():  # definition
    print("hi")  # argument
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
your_name = input("gimme stuff ")  # allows user to enter a type String that is used during runtime
if your_name == "Caroline":
    print("hi nerd") # output = hi nerd if the user types Caroline


# -------------------------------------------------------------------------------------------------------- #

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
dictionary = {"nerd": "caroline", "cool kid": "Tate"} # if you enter the first thing, it gives the second
print(dictionary["cool kid"])  # outputs Tate