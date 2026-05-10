# TODOs

- check what happens to the anomaly maps with no contribution from the 
  patch core.
- separate the background from the actual objects before computing the anomaly,
  a lot of background noise.
- better training and test visualization stats
- greedy subsampling
- grid search
- change ensemble contributions from each branch -> 
  The memory bank produces diffuse, spatially smooth scores — it scores 
  entire patches, not individual pixels. That hurts AP because it lights up 
  large neighbourhoods around true anomalies, inflating false positives. 
  The SegHead is better suited for AP since sigmoid output is spatially 
  sharper and more localized.
- percentage of clean samples in training the synth dataset
- Per-class normalization uses train-only statistics (good images), so the 
  upper bound (99.9th percentile of normal-image scores) might be too low — the 
  test set contains real anomalies that score above anything seen during 
  normalization, and they all get clipped to 1.0. This compresses the tail of 
  the score distribution, which can hurt AP since it relies on fine-grained 
  ranking. Using mixed train + test scores (or just the test scores) for the 
  upper percentile might help.
- test time augmentation
