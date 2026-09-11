# GAN + Diffusion + CNN Hybrid Pipeline (8-class Spectrogram Classification)

1. Input: 8-class RGB spectrogram images
2. **GAN**: trained adversarially on the spectrograms; the Discriminator's
   intermediate layer is reused as a feature extractor (FEATURE_DIM-dim vector
   per image).
3. **Diffusion**: a feature-space DDIM trained to denoise the GAN features;
   at inference, a partial noise + reverse-denoise cycle "refines" each
   feature vector.
4. **CNN**: 1D-conv + dense classifier that maps refined features to 8
   output classes.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters |
| `dataset.py` | `SpectrogramDataset` |
| `gan.py` | Generator, Discriminator, GAN training loop, feature extraction |
| `diffusion.py` | Feature-space DDIM: forward process, training, refinement sampling |
| `cnn.py` | CNN classifier, training loop, evaluation |
| `train.py` | End-to-end orchestration scrip |


## Setup

```bash
pip install torch torchvision scikit-learn pillow numpy
```

## Running the model

1. Confirm `config.DATA_ROOT` points at your data folder:
   ```
   D:\GAN_Diffusion_CNN_Paper_Material\Class_Wise_RGB_Spectrogram_DEAP
   ```
   (already set as the default in `config.py`).
2. Fix the class-folder names if needed. Based on the standard DEAP
   valence/arousal/dominance labeling scheme, the 8 folders should be:
   `HVHAHD, HVHALD, HVLAHD, HVLALD, LVHAHD, LVHALD, LVLAHD, LVLALD`.
3. Run:
   ```bash
   python train.py
   ```

## Quick pipeline sanity check (no real data needed)

```bash
python train.py --dummy --gan_epochs 2 --diff_epochs 2 --cnn_epochs 2
```

## Multi-seed runs (for the statistical validation discussed earlier)

To reproduce the seed-based evaluation setup from your statistical
validation section, run once per seed and record the final test accuracy:

```bash
python train.py --seed 42
python train.py --seed 7
python train.py --seed 123
python train.py --seed 999
python train.py --seed 2024
```

## Notes / things to adapt

- `FEATURE_DIM`  and `IMG_SIZE` in `config.py` can be tuned to
  your actual spectrogram resolution and desired feature dimensionality.
- `DIFF_TIMESTEPS` / `DIFF_REFINE_STEPS` control how strongly the diffusion
  stage refines (denoises) the GAN features — increase `DIFF_REFINE_STEPS`
  for stronger refinement, decrease it to stay closer to the raw GAN
  features.
- The CNN is implemented as Conv1d + dense layers operating on the flat
  refined-feature vector (rather than the original 2D image), since its
  input at this stage is the diffusion-refined *feature vector*, not an
  image.
