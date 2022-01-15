import os
import glob
import shutil
import uuid


def make_valid_dataset():
    for num in range(1, 76):
        files = glob.glob(f'.//{num}/*.jpg')
        os.makedirs(f"./valid-dataset/{num}", exist_ok=True)
        shutil.move(files[0], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')
        shutil.move(files[1], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')
        shutil.move(files[2], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')


def count_dataset(folder='train-dataset'):
    sum = 0
    for num in range(1, 76):
        files = glob.glob(f'./{folder}/{num}/*.jpg')
        sum += len(files)
        print(f"{num}: {len(files)}")
    files = glob.glob(f'./{folder}/x/*.jpg')
    print(f"x: {len(files)}")
    sum += len(files)
    print(f"sum: {sum}")


if __name__ == "__main__":
    # make_valid_dataset()
    count_dataset()
