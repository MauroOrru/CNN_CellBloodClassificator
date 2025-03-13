# CNN for Blood Cell Classification: AComparative Study of ResNet and EfficientNet

## Repository Composition

- **dataset.py**:
  - Defines the `BloodCellDataset` class, which inherits from PyTorch's `Dataset` class.
  - **Key attributes:**
    - `root_dir`: Path to the dataset folder.
    - `valid_folders`: List of valid folder names to load.
    - `img_size`: Dimension for the images (width and height are both set to `img_size`).
    - `transform`: Optional transformations (e.g., data augmentation) to be applied to the images.
