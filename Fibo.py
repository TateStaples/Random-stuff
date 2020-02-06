# october / november 2019

fibs = {0: 0, 1: 1}
def Fibonacci_Sequence(n):
    global fibs
    if n in fibs:
        return fibs[n]

    value = Fibonacci_Sequence(n-1) + Fibonacci_Sequence(n-2)
    fibs[n] = value
    return value

print(Fibonacci_Sequence(998)+ Fibonacci_Sequence(999))
