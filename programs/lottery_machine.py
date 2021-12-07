import random

class LotteryMachine:
    def __init__(self, balls):
        random.shuffle(balls)
        self.balls = balls
        self.balls_drawn = []
        self.balls_left = len(balls)

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
        path = "lottery_result.txt"
        with open(path, "w") as f:
            f.write("\n".join(map(str,self.balls_drawn)))

    def __str__(self):
        return "LotteryMachine(balls={}, balls_drawn={}, balls_left={})".format(
            self.balls, self.balls_drawn, self.balls_left)
def main():
    lotteryMachine = LotteryMachine(list(range(1,76)))
    while(True):
        _ = input()
        try:
            print(lotteryMachine.draw())
        except ValueError:
            print("No more balls!")
            break

if __name__ == '__main__':
    main()
