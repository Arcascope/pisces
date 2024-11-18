# Example data directory for use with `pisces`

In `../examples` you will find code used in publications. For these examples, we draw data from CSVs inside this directory. It looks like this:
``` sh
data
├── walch_et_al
│   ├── cleaned_accelerometer
│   │   ├── 46343_cleaned_motion.out
│   │   ├── 759667_cleaned_motion.out
│   │   ├── ...
│   ├── cleaned_activity
│   │   ├── 46343_cleaned_counts.out
│   │   ├── 759667_cleaned_counts.out
│   │   ├── ...
│   ├── cleaned_psg
│   │   ├── 46343_cleaned_psg.out
│   │   ├── 759667_cleaned_psg.out
│   │   ├── ...
├── hybrid_motion
│   ├── cleaned_accelerometer
│   │   ├── 46343.csv
│   │   ├── 759667.csv
│   │   ├── ...
│   ├── cleaned_activity
│   │   ├── 46343.csv
│   │   ├── 759667.csv
│   │   ├── ...
│   ├── cleaned_psg
│   │   ├── 46343_labeled_sleep.txt
│   │   ├── 759667_labeled_sleep.txt
│   │   ├── ...
```

This is set up to be imported by `pisces`, which will find 2 data sets:
```python
from pisces.data_sets import DataSetObject

DATA_LOCATION = Path("/path/to/pisces/data")  # copy-paste your path here

sets = DataSetObject.find_data_sets(DATA_LOCATION)

print(type(sets))  # dict
print(list(sets.keys()))  # ['walch_et_al', 'hybrid_motion']

print(sets['walch_et_al'].features)  # ['accelerometer', 'activity', 'psg']
```

Features are identified as the directories that start with `cleaned_`; these are expected to contain CSVs using either tab, space, or comma for delimiter, but the filename extension is not important.

Subject ids are extracted by comparing filenames in the same feature. E.g., because all of the files in `data/walch_et_al/cleaned_accelerometer` end with `_cleaned_motion.out`, pisces infers that the subject IDs are the part of the filename before this. It combines these with the subject IDs found in `cleaned_activity/` (which all end with `_cleaned_counts.out` hence that part is stripped for this folder to extract an ID), and so on with `cleaned_psg` (and any other feature directories it finds). 