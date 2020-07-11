from math import *


def get_acceleration(x, y):
    magnitude = sqrt(x**2 + y**2)
    grav = get_gravity(magnitude)
    direction = atan(y/x) if x != 0 else pi/2 if y>=0 else 3*pi/2
    if x < 0:
        direction += pi
    ax = cos(direction) * grav
    ay = sin(direction) * grav
    return ax, ay

def get_gravity(dis):
    return 10

def sign(x):
    return 1 if x >= 0 else -1


t = 144.338
intial_height = 6371000
x_speed = 20 * sqrt(2)
y_speed = 20
step_time = 0.1
time = 0
x = 0
y = intial_height

if __name__ == '__main__':
    tick = 0
    while time < t:
        time += step_time  # move forward in time
        tick += 1
        # move object
        x += x_speed * step_time
        y += y_speed

        # accelerate object
        ax, ay = get_acceleration(x, y)
        x_speed += ax * step_time
        y_speed += ay * step_time

        if x >= 5000:
            print(f"we made it at t={time}")
            break
        if tick <= 10:
            tick = 0
            print(x, y)

    print(x, y)
