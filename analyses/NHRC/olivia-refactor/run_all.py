import time
from preprocess_and_save import do_preprocessing
from train_and_finetune import load_and_train
from make_triplots import create_histogram

if __name__ == "__main__":
    start_time = time.time()

    do_preprocessing()
    load_and_train()
    create_histogram("naive")
    create_histogram("lr")
    create_histogram("finetune")
    end_time = time.time()

    print(f"Everything took {end_time - start_time} seconds")
