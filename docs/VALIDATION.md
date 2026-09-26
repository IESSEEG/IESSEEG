# Release validation

The checks below compare the public interface with the final experiment artifacts.

| Component | Check completed |
| --- | --- |
| Data export | All 950 EDF files: digital signal bytes and calibration unchanged; identity headers and free-text annotations removed; relative timekeeping retained |
| Final benchmark metrics | All 70 diagnosis/response evaluations recomputed from original predictions and matched to the final tables |
| Participant joins | 100 participants; 150 original recordings; 600 selected segments; 200 routine segments; explicit parent links |
| Label scope | State labels only for selected segments; clinician diagnosis labels only for routine segments; 140 BASED scores linked to the 20-recording subset |
| Diagnostic qEEG fitting | All 400 test probabilities across the two feature sets exactly match the final saved probabilities when using the original feature cache |
| Response qEEG fitting | Both visits match the final saved probabilities to floating-point precision when using the original feature cache |
| Section 6 classifiers | All 30 PRE/POST conditions match the saved AUROC, confidence intervals, and balanced accuracy with the original feature cache |
| Signal reader | Native EDF and exported EDF yield identical arrays for a real sample from each subset |
| Model entrypoints | All 16 diagnosis training/inference entrypoints load and accept their CLI arguments |
| Diagnosis preprocessing | One complete routine segment in each of four preprocessing families matches the original cached arrays exactly |
| Diagnostic feature extraction | A complete routine segment yields identical qEEG features before and after EDF sanitization |

These checks distinguish numerical agreement using existing features from a
fresh extraction and full retraining. Five-fold BIOT frozen-head training from the original embeddings exactly
reproduces its 200 saved test probabilities. Full retraining of every neural
model has not been repeated for the release.

All eight response model adapters have also loaded their original pretrained
weights and completed a finite forward pass on real EEG from the public export.
The check covers native preprocessing, embedding dimensions, and prediction
heads; it is not a full training reproduction.

## Internal validation and test partitions

The submitted paper's Appendix B.4 retains model-specific internal validation
for diagnosis full fine-tuning. The release preserves those procedures. BIOT,
CBraMod, and LaBraM split internal training/validation by segment; the other five
diagnostic fine-tuning adapters split by patient. All outer diagnostic test
folds are patient-disjoint. qEEG, frozen diagnosis probes, and both response
benchmarks use separate training, validation, and test patients as specified
by their fixed folds. No internal validation scores should be interpreted as
held-out test performance.

The focused local test suite passes 34 tests. The GitHub workflow runs the
statistical and partition tests without EEG files or model weights.

## BIDS validation of the published dataset

The HF dataset revision `ae9f68fbf10dec84b25c8252826b7e4e7dc4422f`
passes BIDS Validator 3.0.2 with zero errors. The
[dataset validation summary](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/bids_validation.json)
records the remaining warnings. Recommended acquisition fields, anonymized
attribution, and event tables that are unavailable are not invented to suppress warnings.

Duplicate auxiliary `POL RESP` channel names were made unique in 304 EDF headers
and their channel tables. Their original names remain in `source_name`.
Digital samples and physical calibration in all 304 affected files were checked
against their original sources and are unchanged. Scalp EEG labels are unchanged.

The code's `validate` command checks the release inventory, joins, folds and
file availability. It is separate from the official BIDS validator. Benchmark
annotation tables, fixed folds and sampling coordinates are BIDS extensions;
they are excluded from BIDS filename validation and verified by the code adapter.
The updated data successfully prepare all 950 benchmark inputs, including 600
selected-segment rows and 200 routine-segment rows.
