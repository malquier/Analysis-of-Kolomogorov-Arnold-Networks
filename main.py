import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return np.exp(-x**2)


def main():
    x = np.linspace(-5, 5, 200)
    y = list(map(func, x))

    plt.plot(x,y)
    plt.grid(True)
    plt.show()
    return ()
    print("Hello world")

if __name__ == "__main__":
    main()