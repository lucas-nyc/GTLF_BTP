import os

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get("BTP_DATASET_ROOT", os.path.join(PACKAGE_ROOT, "dataset"))
COMPLETE_ROOT = os.path.join(DATASET_ROOT, "complete")

INPUT_DATA = os.path.join(COMPLETE_ROOT, "all_tbl_manual_annotated_v2_cleaned.csv")
LANDMARK_COORDS = os.path.join(COMPLETE_ROOT, "landmark_coordinates_all_cases.csv")
PCA_LOADINGS = os.path.join(COMPLETE_ROOT, "pca_loadings_train_split.xlsx")
PCA_TEMP_CSV = os.path.join(COMPLETE_ROOT, "pca_temp.csv")

IMPUTED_ROOT = os.path.join(DATASET_ROOT, "imputed")  # contains <method>/MNAR and <method>/MCAR
IMPUTED_MNAR_DIR = "MNAR"
IMPUTED_MCAR_DIR = "MCAR"

MISSING_ROOT = os.path.join(DATASET_ROOT, "missing")  # contains MNAR/ and MCAR/
MISSING_MNAR_DIR = "MNAR"
MISSING_MCAR_DIR = "MCAR"

OUT_PATH = os.path.join(PACKAGE_ROOT, "out")

# Imputation methods available under IMPUTED_ROOT/<method>/.
AVAILABLE_IMPUTATION_METHODS = [
    "CMILK", "KNN_Imputation", "Linear_Interpolation",
    "Linear_Regression", "Mean_Imputation", "Moving_Mean", "Moving_Median",
    "GAIN", "MiceForest"
]

# Active imputation methods for baseline evaluation.
IMPUTATION_METHODS = ["CMILK"]

# Regex patterns (can modify if your filenames differ)
MNAR_IMPUTED_REGEX = r"^imputed_dataset_(?:set_)?(?P<idx>\d+)_(?P<scenario>non_occluded|face_mask|glasses)\.csv$"
MCAR_IMPUTED_REGEX = r"^imputed_dataset_set_(?P<idx>\d+)\.csv$"

MISSING_REGEX = r"^missing_dataset(?:_set)?_(?P<idx>\d+)(?:_(?P<scenario>non_occluded|face_mask|glasses)|_percent)?\.csv$"

# Loader params
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.0
TEST_RATIO = 0.3
TRAIN_SIZE = None
# CV folds for train_cv (separate from TRAIN_SIZE).
CV_SPLITS = 95
# If True, bypass K-fold CV and train with one shuffled holdout split.
RUN_WITHOUT_CV = False
# Validation holdout ratio used only when RUN_WITHOUT_CV=True and no fixed
# validation split is supplied to the training routine.
NO_CV_VAL_RATIO = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
THRESHOLD = 0.6            # for PCC edge construction
GPR_THRESHOLD = 0.6        # for GPR edge construction
GPR_N_RESTARTS = 2
USE_TQDM = True
# Guard for unusually large normalized y values used in GPR feature construction.
# MinMax scaling can exceed [0,1] on out-of-range data; keep strict=False unless debugging.
GPR_NORM_GUARD_ABS = 20.0
GPR_NORM_GUARD_STRICT = False
PATIENCE = 10
TARGET = 39
SAVE_RUN_LOG = True
SAVE_CV_METRICS = True

# GraphSAGE+EGNN temperature feature resolution (must map to a real input column).
# Data columns are Var1..Var68; this index is 0-based.
TEMPERATURE_FEATURE_IDX = 0

# Training verbosity
TRAIN_VERBOSE = True
TRAIN_VERBOSE_GNN = True
TRAIN_VERBOSE_VECTOR = False
TRAIN_LOG_EVERY = 5

# GraphSAGE baseline training knobs.
GRAPH_SAGE_EPOCHS = 100
GRAPH_SAGE_LR = 5e-3
GRAPH_SAGE_PATIENCE = 10

# Graph backbone target-focused objective defaults.
GRAPH_TARGET_LOSS_WEIGHT = 1.0
GRAPH_TARGET_ONLY_LOSS = True
GRAPH_LOSS_NAME = "mse"
GRAPH_SMOOTH_L1_BETA = 0.1

BASELINE_VECTOR_BACKBONES = ["linear", "mlp", "cnn1d", "resnet_mlp", "tabnet", "ft_transformer"]
BASELINE_GRAPH_BACKBONES = ["gnn", "graphsage", "gat", "gin"]
BACKBONE_KEYS = BASELINE_VECTOR_BACKBONES + BASELINE_GRAPH_BACKBONES

BACKBONE_DISPLAY = {
    "linear": "LINEAR",
    "mlp": "MLP",
    "cnn1d": "CNN",
    "resnet_mlp": "RESNET_MLP",
    "tabnet": "TABNET",
    "ft_transformer": "FT_TRANSFORMER",
    "gnn": "GNN",
    "graphsage": "GRAPHSAGE",
    "gat": "GAT",
    "gin": "GIN"
}
