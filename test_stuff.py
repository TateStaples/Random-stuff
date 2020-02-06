import math
def works(x, r, b):
    try:
        test1 = x == (3*b + 1) / (pow(r, 3) - 1)
        test2 = b == (x * (pow(r, 3) -1) -1) / 3
        limit = 100
        under100 = x + 3 * b < limit and x * pow(r, 3) < limit
    except ZeroDivisionError:
        return False
    return test1 and test2 and under100

def display_scores(x, r, b):
    buffer = 1
    for i in range(buffer):
        print()
    print("Team \t |\t Q1\t |\t Q2\t |\t Q3\t |\t Q4")
    print("-----------------------------------------------------")
    print(f"Team1 \t | \t{x} \t | \t{x+b} \t | \t{x+b*2} \t | \t{x+b*3}")  # arithmetic
    print("-----------------------------------------------------")
    print(f"Team2 \t | \t{x} \t | \t{x*r} \t | \t{x*r*r} \t | \t{x*r*r*r}")  # arithmetic
    for i in range(buffer):
        print()

for x in range(100):
    for r in range(100):
        for b in range(100):
            if works(x, r, b):
                display_scores(x, r, b)
                #final_score = x+3*b
                #print(f"x: {x}, r: {r}, b: {b}, scores: {(final_score, final_score+1)}, At half: {(x + b, x*r)}")