import random
import string
from typing import List


item_names = [
    "Phone",
    "Laptop",
    "TV",
    "Shoe",
    "Watch",
    "Ball",
    "Toy",
    "Cable",
    "Charger",
    "Book",
    "Headphone",
    "Cup",
    "Jar",
    "Glasses",
    "Candy",
    "Cover",
    "Flash",
    "Harddisk",
    "Pen",
    "Pencil",
]


def generate_item_names(n_items):
    item_names = []

    for i in range(n_items):
        name = ""

        while True:
            name = string.ascii_uppercase[i % 26] + name
            i //= 26

            if i == 0:
                break

        item_names.append(name)

    return item_names


def generate_items(
    filename: str,
    n_item_types: int = 10,
    capacity: float = 50,
    min_item_weight: float = 1,
    max_item_weight: float = 20,
    min_item_value: float = 1,
    max_item_value: float = 100,
    min_item_count: int = 1,
    max_item_count: int = 5,
    item_names: List[str] = item_names,
):
    item_names = item_names[:n_item_types]

    items = []

    for item_name in item_names:
        item_weight = random.uniform(min_item_weight, max_item_weight)
        item_count = random.randint(min_item_count, max_item_count)
        item_value = random.uniform(min_item_value, max_item_value)
        items.append(f"{item_name}, {item_weight:.2f}, {item_value:.2f}, {item_count}")

    with open(filename, "w") as f:
        f.write(f"{capacity}\n")
        f.write("item, weight, value, n_items\n")
        f.write("\n".join(items))


if __name__ == "__main__":
    generate_items("10_items.txt", n_item_types=10)
    generate_items("15_items.txt", n_item_types=15)
    generate_items("20_items.txt", n_item_types=20)
