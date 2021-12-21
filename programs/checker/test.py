import random
from main import Checker


def make_bingo_card():
    bingo_card = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            bingo_card[i][j] = random.randint(1 + i * 1, 15 + 15 * i)
    return bingo_card


def test():
    checker = Checker()
    bingo_card = make_bingo_card()
    for _ in range(40):
        bingo_number = random.randint(1, 75)
        checker.add_output_bingo_number(bingo_number)
    checker.print_output_bingo_numbers()
    checker.print_bingo_card(bingo_card)
    if checker.check_bingo(bingo_card):
        print("BINGO!")
    else:
        print("NOT BINGO!")


if __name__ == "__main__":
    test()
