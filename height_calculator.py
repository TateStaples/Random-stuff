import math

height = input("what is the starting height?")
int_vel = input("what is the initial upward vel")
acc_of_gravity = input("What is the acceleration of gravity?")
time_of_0 = input("When did it reach height 0?")

# solve for height
if height == "?":
    int_vel = int(int_vel)
    acc_of_gravity = int(acc_of_gravity)
    time_of_0 = int(time_of_0)

    dis_fallen = int_vel * time_of_0 - acc_of_gravity * time_of_0**2

    print("The initial height is", dis_fallen * -1)

elif int_vel == "?":
    height = int(height)
    acc_of_gravity = int(acc_of_gravity)
    time_of_0 = int(time_of_0)

    # -H = vt - at^2

    int_vel = (height - acc_of_gravity * time_of_0 ** 2) / time_of_0
    print("the initial velocity was", int_vel)

elif acc_of_gravity == "?":
    int_vel = int(int_vel)
    height = int(height)
    time_of_0 = int(time_of_0)

else:
    height = int(height)
    acc_of_gravity = int(acc_of_gravity)
    int_vel = int(int_vel)

    # use quadratic equation
    hit_ground = (-int_vel + math.sqrt(int_vel**2 + 2 * acc_of_gravity * height)) / 2 * acc_of_gravity

    print("you reached zero at", hit_ground)
