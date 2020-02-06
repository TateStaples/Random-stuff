import random

alphabet = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x',
                'c', 'v', 'b', 'n', 'm',
                '.', '.', ',', '?', '!',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                '/', '-', '–', ' ']


def letter_swap():
    paper = open("original.txt")
    encrypted = open("new.txt", 'w')
    unchosen = alphabet.copy()
    new_letter = {}
    for letter in alphabet:
        new = random.choice(unchosen)
        new_letter[letter] = new
        unchosen.remove(new)
    print(new_letter)
    for line in paper:
        new_line = ""
        #line = line.strip()
        for letter in line:
            letter = letter.lower()
            if letter not in alphabet:
                new_line += letter
                continue
            new_line += new_letter[letter]
        new_line += "\n"
        encrypted.write(new_line)
    encrypted.write(str(new_letter))


def letter_multiplication():
    paper = open("original.txt")
    encrypted = open("new.txt", 'w')
    for line in paper:
        line = line.strip()
        char_at = 0
        new_line = ""
        for letter in line:
            char_at += 1
            if letter not in alphabet:
                new_line += letter
            else:
                new_letter = alphabet[char_at * alphabet.index(letter)%len(alphabet)]
                new_line += new_letter
        encrypted.write(new_line)
        encrypted.write("\n")


def to_bytes():
    paper = open("original.txt")
    encrypted = open("new.txt", 'w')
    for line in paper:
        new_line = ""
        for char in line:
            ascii_val = ord(char)
            binary = bin(ascii_val)
            new_line += binary
        encrypted.write(new_line)
        encrypted.write("\n")

letter_swap()