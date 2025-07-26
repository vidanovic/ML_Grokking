# lr_1.py

import numpy as np
import random
import matplotlib.pyplot as plt

from lib.linear_regression import square_trick
from lib.error_estimator import root_mean_square_error


def linear_regression(
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 1000
):

    price_per_room = random.random()
    base_price = random.random()
    rmse_history = []

    for epoch in range(epochs):
        i = random.randint(0, len(labels) - 1)
        num_rooms = features[i]
        price = labels[i]
        price_per_room, base_price = square_trick(
            base_price=base_price,
            price_per_room=price_per_room,
            num_rooms=int(num_rooms),
            price=float(price),
            learning_rate=learning_rate
        )

        # Calculate predictions and RMSE for all data
        predictions = price_per_room * features + base_price
        rmse_history.append(root_mean_square_error(labels=labels, predictions = predictions))

    return price_per_room, base_price, rmse_history


def main():
    num_of_rooms = np.array([1, 2, 3, 5, 6, 7])
    prices = np.array([155, 197, 244, 356, 407, 448])

    price_per_room, base_price, rmse_history = linear_regression(
        features = num_of_rooms,
        labels = prices,
        learning_rate=0.01,
        epochs=2_000)

    plt.plot(rmse_history)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over epochs')
    plt.show()

    plt.scatter(num_of_rooms, prices, color='blue', label='Data points')

    # Add regression line
    x_line = np.linspace(num_of_rooms.min(), num_of_rooms.max(), 100)
    y_line = price_per_room * x_line + base_price
    plt.plot(x_line, y_line, color='black', label='Regression line')

    plt.xlabel('Number of rooms')
    plt.ylabel('Price')
    plt.title(f'Price per room: {round(price_per_room, 3)} \nBase price: {round(base_price, 3)}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
