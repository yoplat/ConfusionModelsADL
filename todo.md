# TODOs

- separate the background from the actual objects before computing the anomaly,
  a lot of background noise.
- grid search
- Per-class normalization uses train-only statistics (good images), so the 
  upper bound (99.9th percentile of normal-image scores) might be too low — the 
  test set contains real anomalies that score above anything seen during 
  normalization, and they all get clipped to 1.0. This compresses the tail of 
  the score distribution, which can hurt AP since it relies on fine-grained 
  ranking. Using mixed train + test scores (or just the test scores) for the 
  upper percentile might help.
- test time augmentation
