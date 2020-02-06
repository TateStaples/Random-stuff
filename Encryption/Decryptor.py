def un_letter_swap(key):
    encryption = {}
    for swap in key:
        encryption[key[swap]] = swap
    encrypted = open("new.txt")
    for line in encrypted:
        new_line = ""
        for letter in line:
            if letter not in encryption:
                new_line += letter
            else:
                new_line += encryption[letter]
        print(new_line)


def un_letter_multiply():
    encrypted = open("new.txt")
    alphabet = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x',
                'c', 'v', 'b', 'n', 'm',
                '.', '.', ',', '?', '!',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                '/', '-', '–', ' ']
    for line in encrypted:
        char_at = 0
        new_line = ""
        for letter in line:
            char_at += 1
            if letter not in alphabet:
                new_line += letter
            else:
                product = alphabet.index(letter)
                for char in alphabet
                    if product == char_at * char % len(alphabet):
                        


                # new_letter = alphabet[char_at * alphabet.index(letter) % len(alphabet)]

