from time import time
if __name__ == '__main__':
    count = int(input("what do you want to count to?   "))
    # t = time()
    # print(f"Base time: {time()-t}")
    t = time()
    i = 0
    while i < count:
        i += 1
    print(f"count time: {time()-t}")
