"""Every experimental parameter of the project lives here so runs are reproducible."""

SEED = 20260913

# Image preprocessing: images keep their aspect ratio and are downscaled so the longest
# side is at most MAX_SIDE pixels (INTER_AREA for images, nearest neighbour for masks).
MAX_SIDE = 256

# Image-level split fractions (train, validation, test).
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)

# K-Means segmentation.
CHANNELS = {"R": 0, "G": 1, "B": 2}
COMBINATIONS = ["R", "G", "B", "RG", "RB", "GB", "RGB"]
KMEANS_PARAMS = {
    "n_clusters": 2,          # fruit vs background
    "init": "k-means++",
    "n_init": 10,             # restarts; the lowest-inertia solution is kept
    "max_iter": 300,
    "tol": 1e-4,              # relative tolerance on centroid shift (Frobenius norm)
}
PIXEL_SAMPLE = 12000          # pixels used to fit the centroids; all pixels are then assigned

# Morphological post-processing applied to the K-Means mask (evaluated, not assumed).
MORPH_KERNEL = 5              # elliptical structuring element (pixels)
MIN_COMPONENT_FRACTION = 0.01 # connected components smaller than 1 % of the image are removed

# Fruit descriptors computed inside the segmented region.
FEATURES = ["R", "G", "B", "H", "S", "V", "a*", "b*"]

# Analysis-based selection.
ANALYSIS_MAX_FEATURES = 3
ANALYSIS_MAX_ABS_CORR = 0.85

# Sequential forward selection (SFS).
SFS_FOLDS = 5
SFS_REPEATS = 10
SFS_TOLERANCE = 1e-3          # smallest subset whose CV-AUC is within this of the best

# PCA.
PCA_VARIANCE_TARGET = 0.95

# Uncertainty / robustness analyses.
N_BOOTSTRAP = 2000
N_REPEATED_SPLITS = 30

# Pixel sampling for exploratory histograms.
EDA_PIXELS_PER_REGION = 3000
