import polars as pl

DATA_DIR = '/home/eric/Engineering/Work/pisces/data'
FEATURE_COLS = ['ACC_X', 'ACC_Y', 'ACC_Z']
TIMESTAMP_COL = 'TIMESTAMP'
LABEL_COL = 'Sleep_Stage'
NEW_LABEL_COL = 'PSG'
SELECT_COLS = [TIMESTAMP_COL, *FEATURE_COLS, LABEL_COL]
TIMESTAMP_HZ = 64
TIMESTAMP_DT = 1/TIMESTAMP_HZ

MASK_VALUE = -1

ACC_MAX_IDX = 2 ** 22
PSG_MAX_IDX = 33_000

LABEL_MAP = {
    'W': 0,
    'N1': 1,
    'N2': 1,
    'N3': 2,
    'R': 3,
    'Missing': MASK_VALUE,
    'P': 0
}
mapping_df = pl.DataFrame({
    LABEL_COL: list(LABEL_MAP.keys()),
    NEW_LABEL_COL: list(LABEL_MAP.values())
})


