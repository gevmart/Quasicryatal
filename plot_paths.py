from potential import *
import matplotlib.pyplot as plt

# movement = np.array((400, 2))
# for t in np.arange(400):
#     movement[t] = square_movement(0, cutoff=100)
size = 30
cutoff = 1000
# max = cutoff / potential_change_speed / 2 / np.pi
# first = np.concatenate([max / size * np.arange(size), np.repeat(0, size)]).reshape(2, size).T
# second = np.concatenate([np.repeat(max, size), max / size * np.arange(size)]).reshape(2, size).T
# third = np.concatenate([max / size * (size - np.arange(size)), np.repeat(max, size)]).reshape(2, size).T
# fourth = np.concatenate([np.repeat(0, size), max / size * (size - np.arange(size))]).reshape(2, size).T
# movement = -np.concatenate([first, second, third, fourth])

l1, l2 = 0, 1

movement = np.zeros((4 * size, 2))
for t in np.arange(4 * size):
    ps = np.array(semicircle(t, cutoff=cutoff, lasers=(l1, l2)))
    defaults = default_phases()
    xx = ((ps[l1, :] - defaults[l1, :]) / k)[0, 0]
    yy = ((ps[l2, :] - defaults[l2, :]) / k)[0, 0]
    # xx = t
    # yy = 2 * t
    movement[t] = np.array([-xx, -yy])


print(movement)

colors = ['red', 'blue', 'orange', 'black', 'green', 'purple']
fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=18)


plt.quiver(movement[:-1, 0], movement[:-1, 1], movement[1:, 0] - movement[:-1, 0],
               movement[1:, 1] - movement[:-1, 1],
               scale_units='xy', angles='xy', scale=1, color=colors[0])
plt.scatter(0, 0, marker="D", s=100)

plt.title("The path of the modulation", fontsize=18)
plt.xlabel(r"$x/\lambda$", fontsize=18)
plt.ylabel(r"$y/\lambda$", fontsize=18)
plt.show()


plt.show()
