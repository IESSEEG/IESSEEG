# EEGPT adapter

EEGPT uses four-second windows from 19 scalp channels, resampled from 200 to
250 Hz. Diagnosis inputs use exponential moving standardization on the
continuous segment before windowing.

The adapter uses Braindecode 1.7.0 in a separate interpreter. Follow the
[EEGPT setup instructions](../../../docs/MODELS.md#eegpt) to configure the
interpreter and download the weights.

Braindecode's channel projection maps the input montage to its 19-channel
vocabulary. The checkpoint's 62-entry `chans_id` buffer is excluded when
loading weights; the adapter retains the buffer for the input channels and
checks the remaining checkpoint keys.
