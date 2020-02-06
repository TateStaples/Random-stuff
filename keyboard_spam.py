import keyboard
from time import sleep

if __name__ == '__main__':
    sleep(5)
    for i in range(10):
        keyboard.write("Hi Tyler, I am spamming you")
        keyboard.press_and_release("enter")
