import os
import glob
import shutil
import uuid

for num in range(1, 76):
    files = glob.glob(f'./train-dataset/{num}/*.jpg')
    os.makedirs(f"./valid-dataset/{num}", exist_ok=True)
    shutil.move(files[0], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')
    shutil.move(files[1], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')
    shutil.move(files[2], f'./valid-dataset/{num}/{uuid.uuid4()}.jpg')
