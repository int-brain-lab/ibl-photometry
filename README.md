# photometry-tools
A collection of methods and tools for experimental photometry data

## Note about the preferred formats
A good practice is to keep the raw photometry data in a dataframe with columns:
- times
- raw_isosbestic
- raw_calcium
The preferred interchange format is the parquet format (`.pqt`), which is a binary format that is fast to read and write, compressed and keeps typing information.


cf.
ibl-photometry/src/examples/csv_preprocessing.py

