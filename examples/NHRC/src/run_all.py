import time

from examples.NHRC.src.make_triplots import create_histogram
from examples.NHRC.src.preprocess_and_save import do_preprocessing
from examples.NHRC.src.train_and_finetune import load_and_train

if __name__ == "__main__":
    start_time = time.time()

    do_preprocessing()
    # load_and_train()
    # create_histogram("naive")
    create_histogram("lr")
    create_histogram("finetune")
    end_time = time.time()

    print(f"Everything took {end_time - start_time} seconds")
