import random


class LotteryMachine:
    FILE_PATH = "lottery_result.txt"

    def __init__(self,):
        self.balls = random.sample(list(range(1, 76)), 75)
        self.balls_drawn = []
        self.balls_left = 75

        with open(self.FILE_PATH, "w") as f:
            f.write("")

    def draw(self):
        if self.balls_left == 0:
            raise ValueError("No balls left!")
        ball = self.balls[0]
        self.balls_drawn.append(ball)
        self.balls_left -= 1
        self.balls = self.balls[1:]
        self.save()
        return ball

    def save(self):

        with open(self.FILE_PATH, "w") as f:
            f.write("\n".join(map(str, self.balls_drawn)))

    def __str__(self):
        return "LotteryMachine(balls={}, balls_drawn={}, balls_left={})".format(
            self.balls, self.balls_drawn, self.balls_left)


def main():
    lottery_machine = LotteryMachine()
    while(True):
        _ = input("Press enter to get a number")
        try:
            print(lottery_machine.draw())
        except ValueError:
            print("No more balls!")
            break


if __name__ == '__main__':
    main()
