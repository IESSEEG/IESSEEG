# Signal preprocessing kernels

These are the original model input transformations retained by the public
`python -m iesseeg preprocess` command. Use that command to select the correct
subset-specific settings and output location.

- `biot_preprocessing.py`: 18 bipolar channels for BIOT.
- `cbramod_preprocessing.py`: 19-channel referential inputs, with separate
  filtering settings for CBraMod and the LaBraM input family.
- `preprocessing.py`: 22 bipolar channels for LUNA.

Dataset metadata, mappings, and fixed partitions are read by `iesseeg.dataset`
from the published data tables. No private source spreadsheets are needed.
