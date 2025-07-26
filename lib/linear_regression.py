from typing import Tuple


def square_trick(
        base_price: float,
        price_per_room: float,
        num_rooms: int,
        price: float,
        learning_rate: float) -> Tuple[float, float]:

    predicted_price = base_price + price_per_room * num_rooms
    base_price += learning_rate * (price - predicted_price)
    price_per_room += learning_rate * num_rooms * (price - predicted_price)

    return price_per_room, base_price


def absolute_trick(
        base_price: float,
        price_per_room: float,
        num_rooms: int,
        price: float,
        learning_rate: float) -> Tuple[float, float]:

    predicted_price = base_price + price_per_room * num_rooms
    if price >= predicted_price:
        price_per_room += learning_rate * num_rooms
        base_price += learning_rate
    else:
        price_per_room -= learning_rate * num_rooms
        base_price -= learning_rate

    return price_per_room, base_price
