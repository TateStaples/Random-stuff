from math import *
import matplotlib.pyplot as plt


initial_velocity = 5
delta_y = 0
gravity = 9.81


def get_resistance(vel, k):
    return -k*vel

def get_dis(angle, k):
    dt = 0.00001
    y = 0
    x = 0
    x_vel = cos(radians(angle)) * initial_velocity
    y_vel = sin(radians(angle)) * initial_velocity
    while y_vel >= 0 or y >= delta_y:

        # apply gravity
        y_vel -= gravity * dt

        # apply resistance
        x_vel += get_resistance(x_vel, k) * dt
        y_vel += get_resistance(y_vel, k) * dt

        # move
        x += x_vel * dt
        y += y_vel * dt


        # print(x, y)
        # print(x_vel)
    return x


def net_vel(x, y):
    return sqrt(x*x + y*y)


def best_dis(k, last):
    max_dis = None
    best_angle = None
    for a in range(round(last*10-5), round(last*10)) if last is not None else range(0, 30):
        angle = a/10 if last is not None else a * 3
        # print(f"angle: {angle}")
        x = get_dis(angle, k)
        if max_dis is None or x > max_dis:
            max_dis = x
            best_angle = angle
    return best_angle


if __name__ == '__main__':
    domain = -1, 1
    step = .01
    data = []
    k = domain[0]
    ks = []
    while k <= domain[1]:
        print(f"k: {k}")
        ks.append(k)
        data.append(best_dis(k, data[-1] if len(data)> 0 else None))
        k += step
    plt.plot(ks, data)
    plt.show()