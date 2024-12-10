

from examples.NHRC.src.preprocess_and_save import do_preprocessing
from examples.RGB_Spectrograms.preprocessing import big_specgram_process


if __name__ == '__main__':
    import argparse

    do_preprocessing(big_specgram_process)