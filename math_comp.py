from math import *


def is_prime(x):
    max = sqrt(x)
    for i in range(2, int(max)+1):
        div = x/i
        if div == int(div):
            return False
    return True

def get_factors(x):
    factors = []
    max = sqrt(x)
    for i in range(2, int(max) + 1):
        div = x / i
        if div == int(div):
            factors.append(i)
    return factors

def relatively_prime(x, y):
    yfacts = get_factors(y)
    for fx in get_factors(x):
        if fx in yfacts:
            return False
    return True

# number 28:
# for p in range(500):
#     if is_prime(p):
#         print(p)
#         for q in range(500):
#             if is_prime(q):
#                 for r in range(500):
#                     if is_prime(r) and 2*r*q*p + r + q + p == 2020:
#                         print(f"Answer: {p*q + q*r + r*p}")
#                         print(f"p: {p}, q: {q}, r: {r}")


output_lists = set()
def func(l1, l2, l3):
    stuff = True
    if len(l1) > 0:
        stuff = False
        x = l1[0]
        func(l1[1:len(l1)], l2, l3+[x])
    if len(l2) > 0:
        stuff = False
        x = l2[0]
        func(l1, l2[1:len(l2)], l3 + [x])
    if stuff:
        output_lists.add(tuple(l3))
# list1 = list(range(1, 9))
# list2 = list(range(1, 6))
# func(list1, list2, [])
# print(len(output_lists))

# number 13
# def formula(n):
#     return n/(n*n - 1) - 1/n
# sum = sum([formula(i) for i in range(2, 101)])
# print(sum)

# def mean(stuff):
#     stuff = sorted(stuff)
#     return sum(stuff)/len(stuff)
# def median(stuff):
#     stuff = sorted(stuff)
#     return mean(stuff[2:4])
#
# base = [107, 122, 127, 137, 152]
# for i in range(1000):
#     new_stuff = base + [i]
#     if median(new_stuff) == mean(new_stuff):
#         print(i)
# output_line = set()
# docs = [0 for i in range(3)]
# nurses = [1 for i in range(4)]
# patients = [2 for i in range(3)]
# def line_thing(docs, nurses, patients, line, previous_patient=None):
#     did_thing = False
#     if previous_patient != 2 and len(patients) > 0:
#         did_thing = True
#         line_thing(docs, nurses, patients[1:len(patients)], line + [patients[0]], patients[0])
#     if len(docs) > 0:
#         did_thing = True
#         line_thing(docs[1:len(docs)], nurses, patients, line + [docs[0]], docs[0])
#     if len(nurses) > 0:
#         did_thing = True
#         line_thing(docs, nurses[1:len(docs)], patients, line + [nurses[0]], nurses[0])
#     if not did_thing and len(patients) == 0:
#         output_line.add(tuple(line))

def is_gud(a, b, c):
    try:
        val = a * pow(a/b, 1/3) + b * pow(b/c, 1/3) + c * pow(c/a, 1/3)
        return abs(val) < 0.1
    except:
        return False

def loop(low, high, step):
    list = []
    num = low
    while num < high:
        list.append(num)
        num += step
    return list

def thingy(a, b, c):
    return (a**3/b*b*c + b**3/c*c*a + c**3/a*a*b)**2

max = 0
for a in loop(-10, 50, 0.1):
    print(a)
    for b in loop(-10, 50, 0.1):
        for c in loop(-10, 50, 0.1):
            if is_gud(a, b, c):
                val = thingy(a, b, c)
                if val > max:
                    max = val

if __name__ == '__main__':
    print(max)