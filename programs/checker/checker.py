class Checker:
    """
    BINGO CHECKER

    Attributes:
    output_bingo_numbers (list): list of bingo numbers which already outputted in the game
    """

    def __init__(self):
        self.output_bingo_numbers = []

    def import_output_bingo_number(self) -> None:
        ls = ["FREE"]
        with open("./checker/lottery_result.txt") as f:
            for line in f:
                ls.append(int(line))
        self.output_bingo_numbers = ls

    def check_bingo(self, bingo_card):
        self.print_bingo_card(bingo_card)
        self.import_output_bingo_number()
        self.print_output_bingo_numbers()
        binary_bingo_card = self.make_binary_bingo_card(bingo_card)
        self.print_bingo_card(binary_bingo_card)
        if self.check_binary_bingo_card(binary_bingo_card):
            return True
        return False

    def make_binary_bingo_card(self, bingo_card):
        binary_bingo_card = [[0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if bingo_card[i][j] in self.output_bingo_numbers:
                    binary_bingo_card[i][j] = 1
        return binary_bingo_card

    def check_binary_bingo_card(self, binary_bingo_card):
        for i in range(5):
            if self.check_row(binary_bingo_card, i):
                return True
        for i in range(5):
            if self.check_column(binary_bingo_card, i):
                return True
        if self.check_diagonal(binary_bingo_card):
            return True
        if self.check_anti_diagonal(binary_bingo_card):
            return True
        return False

    def check_row(self, binary_bingo_card, row_index):
        row = binary_bingo_card[row_index]
        if row.count(1) == 5:
            return True
        return False

    def check_column(self, binary_bingo_card, column_index):
        column = [row[column_index] for row in binary_bingo_card]
        if column.count(1) == 5:
            return True
        return False

    def check_diagonal(self, binary_bingo_card):
        diagonal = [binary_bingo_card[i][i] for i in range(5)]
        if diagonal.count(1) == 5:
            return True
        return False

    def check_anti_diagonal(self, binary_bingo_card):
        anti_diagonal = [binary_bingo_card[i][4 - i] for i in range(5)]
        if anti_diagonal.count(1) == 5:
            return True
        return False

    def print_bingo_card(self, bingo_card_list):
        for row in bingo_card_list:
            for col in row:
                print("{:^6s}".format(str(col)), end=" ")
            print()
        print()

    def print_output_bingo_numbers(self):
        print(self.output_bingo_numbers)
        print()


if __name__ == '__main__':
    pass
