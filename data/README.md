# Dataset Setup

This repository includes `labels.csv`, but it does **not** include `images.npy`.

The original image array is approximately **495 MB**, which makes it unsuitable for a normal GitHub repository.

## Expected files

Place your local copy of the image array here:

```text
data/
├── images.npy
└── labels.csv
```

The project expects:

```text
images.npy shape: (4125, 200, 200, 3)
labels.csv rows: 4125
```

The labels used in the notebook are:

```text
0 = Without Helmet
1 = With Helmet
```

Class counts:

```text
Without Helmet: 964
With Helmet: 3161
```

## Google Colab

One simple approach is:

1. Upload this repository to Google Drive or clone it in Colab.
2. Copy `images.npy` into the repository's `data/` folder.
3. Run the notebook from the repository root.

Example:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then either work directly from a Drive folder or copy the dataset into the Colab working directory.

## Git LFS

If you own the dataset and are permitted to distribute it, Git LFS is another option for large binary files. For a public portfolio repository, keeping the large dataset outside the repository is usually cleaner.
