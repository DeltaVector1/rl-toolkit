# MXFP4 archive

This folder preserves the retired GLM-5.2 MXFP4 work. It is not used by the
current FP8 run.

- `patches/prime-rl-mxfp4.patch`: complete local PrimeRL changes against commit
  `f50ec5c4c8c45864ec535dead2c3b630de50d068`
- `mxfp4.py`: standalone packed-weight reader and dequantization code
- `configs/`: final working run settings and earlier trainer settings
- `metadata/`: model configuration, Hub metadata, and weight index

The step-6 LoRA adapter and its metrics remain on n82 in the final MXFP4 run
folder. Model weights and optimizer checkpoints are not copied here.
