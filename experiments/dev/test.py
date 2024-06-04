from tqdm import tqdm
import time


if __name__ == '__main__':
    for i in tqdm(range(100), desc="Outer", position=1):
        for j in tqdm(range(100), desc="Inner", position=0, leave=False):
            time.sleep(0.1)
