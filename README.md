# MNIST Digit Classification

> **Status:** Partially completed  
> **Last updated:** 2026-11-18

## Overview

MNIST digit classification experiments with a fully connected network and a CNN. The main workflow is in [`notebook.ipynb`](notebook.ipynb).

## Project files

- `notebook.ipynb` — training, evaluation, and predictions.
- `hyperparams.json` — CNN configurations.
- `src/helper.py` — custom image dataset and prediction plots.
- `src/util.py` — loss and plotting helpers.
- `models/` — saved model checkpoints and results.
- `my_test_set/` — personal PNG test images.

## Setup

```bash
python -m pip install -r requirements.txt
```

Open the notebook and run cells from top to bottom. MNIST downloads to `src/data` if needed.

## Models and MNIST results

We began with a fully connected neural network (FCNN): flatten 28×28 pixels → 128-unit ReLU layer → 10 output classes. Its latest saved test accuracy was **96.76%**.

Then we trained a CNN: two 3×3 convolution layers (24 then 36 channels), each followed by ReLU and 2×2 max pooling; then a 128-unit fully connected layer and 10 outputs. Each run used batch size 128 for 7 epochs.

CNN runs, copied from the notebook's sorted results table:

| Rank | `train_loss` | `test_loss` | `train_accuracy` | `test_accuracy` | `batch_size` | `epochs` | `lr` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.017502 | 0.000217 | 0.993400 | 0.9904 | 128 | 7 | 0.20 |
| 2 | 0.000559 | 0.000243 | 0.993950 | 0.9889 | 128 | 7 | 0.30 |
| 3 | 0.012732 | 0.000265 | 0.989400 | 0.9888 | 128 | 7 | 0.10 |
| 4 | 0.038171 | 0.000346 | 0.994117 | 0.9870 | 128 | 7 | 0.50 |
| 5 | 0.050058 | 0.000385 | 0.984417 | 0.9844 | 128 | 7 | 0.05 |
| 6 | 0.061931 | 0.000523 | 0.977833 | 0.9791 | 128 | 7 | 0.03 |
| 7 | 0.075766 | 0.000692 | 0.986750 | 0.9729 | 128 | 7 | 0.07 |
| 8 | 0.050558 | 0.000694 | 0.967667 | 0.9717 | 128 | 7 | 0.40 |
| 9 | 0.126115 | 0.001063 | 0.953717 | 0.9595 | 128 | 7 | 0.01 |

The best MNIST run used learning rate **0.20** and reached **99.04%** test accuracy.

## Test personal images

1. Put labeled PNGs in the project-root `my_test_set/` folder.
2. Keep the current path-based label format (for example, `My-2.png`). Label parsing is in `MyImageDataset` in `src/helper.py`.
3. Run the notebook's final section. It evaluates all saved models and writes prediction grids and `errors.json` files under `models/`.

On the current 27 images, the saved custom-set results were:

| Rank by MNIST accuracy | Learning rate | Errors | Accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.20 | 2 / 27 | 92.59% |
| 2 | 0.30 | 2 / 27 | 92.59% |
| 3 | 0.10 | 6 / 27 | 77.78% |
| 4 | 0.50 | 2 / 27 | 92.59% |
| 5 | 0.05 | 4 / 27 | 85.19% |
| 6 | 0.03 | 3 / 27 | 88.89% |
| 7 | 0.07 | 5 / 27 | 81.48% |
| 8 | 0.40 | 5 / 27 | 81.48% |
| 9 | 0.01 | 6 / 27 | 77.78% |

The strongest CNN runs reached **92.59%** on the custom images, correctly classifying 25 of 27. That is a strong result on personally created data and suggests the models generalize well to this sample.

### Image preprocessing

- The notebook applies grayscale, resize, tensor conversion, and per-image max scaling.
- `MyImageDataset.__getitem__` calls `normalize_digit` in `src/helper.py`, which thresholds, crops, resizes, and centers the digit.
- This differs from MNIST preprocessing and can shift contrast or alignment, contributing to custom-set errors.
- For implementation details, see the transform definition in the notebook and `normalize_digit` in `src/helper.py`.

## Notes

- The printed CNN epoch loss is always zero because its accumulator is not updated.
- Saved `test_loss` values are under-scaled by the evaluation code; the accuracy columns are unaffected.
- Replace path-based label parsing with explicit labels when convenient.

## License

This project is licensed under the [MIT License](LICENSE).
