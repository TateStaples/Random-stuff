# october 2019

def englishize(x):
    prefixes = {0: "", 1: "un", 2: "duo", 3: "tri", 4: "quadr", 5: "quint", 6: "sext", 7: "sept", 8: "oct", 9: "non",
                10: "dec", 20: "vigint", 30: "trigint", 40: "quadragint", 50: " quinquagint",
                60: "sexagint", 70: "septuagint", 80: "octogint", 90: "nonagint"}
    words = {0: "", 1: "thousand", 2: "million", 3: "billion"}

    ones = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    teens = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
             16: "sixteen", 17: "seventeen", 18: "eight", 19: "nineteen"}
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

    string = str(x)
    sets = []
    for i in range(len(string)//3+1):
        index = -3*i
        set = string[index-3: index]
        if index == 0:
            set = string[-3:]
        if len(set) == 0: continue
        sets.append((set, i))

    for nums, count in reversed(sets):
        word = ""
        for index, digit in enumerate(nums):
            if digit != "0":
                if index == 0:  # 100s
                    print(ones[int(digit)], "hundred", end=" ")
                elif index == 1:  #10s
                    if digit == "1":  # teens
                        print(teens[int(nums[1:])], end=" ")
                        break
                    print(tens[int(digit)], end=" ")
                else:
                    print(ones[int(digit)], end=" ")
            if count in words:
                print(word + words[count], end=" ")
            else:
                tens_place = prefixes[int(count) // 10 * 10]
                ones_place = prefixes[int(count) % 10]
                print(word + ones_place + tens_place + "illion", end=" ")


englishize(5536354663456)
