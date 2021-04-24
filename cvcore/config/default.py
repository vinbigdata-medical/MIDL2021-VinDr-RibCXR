from yacs.config import CfgNode as CN


# Create root config node
_C = CN()
# Config name
_C.NAME = ""
# Config version to manage version of configuration names and default
_C.VERSION = "0.1"


# ----------------------------------------
# System config
# ----------------------------------------
_C.SYSTEM = CN()

# Number of workers for dataloader
_C.SYSTEM.NUM_WORKERS = 8
# Use GPU for training and inference. Default is True
_C.SYSTEM.CUDA = True
# Random seed for seeding everything (NumPy, Torch,...)
_C.SYSTEM.SEED = 0
# Use half floating point precision
_C.SYSTEM.FP16 = True
# FP16 Optimization level. See more at: https://nvidia.github.io/apex/amp.html#opt-levels
_C.SYSTEM.OPT_L = "O2"


# ----------------------------------------
# Directory name config
# ----------------------------------------
_C.DIRS = CN()

# Train, Validation and Testing image folders
_C.DIRS.TRAIN_IMAGES = ""
_C.DIRS.VALIDATION_IMAGES = ""
_C.DIRS.TEST_IMAGES = ""
# Trained weights folder
_C.DIRS.WEIGHTS = "./weights/"
# Inference output folder
_C.DIRS.OUTPUTS = "./outputs/"
# Training log folder
_C.DIRS.LOGS = "./logs/"


# ----------------------------------------
# Datasets config
# ----------------------------------------
_C.DATA = CN()
# Create small subset to debug
_C.DATA.DEBUG = False
# Datasets problem (multiclass / multilabel)
_C.DATA.TYPE = ""
# Image size for training
_C.DATA.IMG_SIZE = (224, 224)
# Image input channel for training
_C.DATA.INP_CHANNEL = 3
# For CSV loading dataset style
# If dataset is contructed as folders with one class for each folder, see ImageFolder dataset style
# Train, Validation and Test CSV files
_C.DATA.JSON = CN()
_C.DATA.JSON.TRAIN = ""
_C.DATA.JSON.VAL = ""
_C.DATA.JSON.TEST = ""

# ----------------------------------------
# Training config
# ----------------------------------------
_C.TRAIN = CN()

# Number of training cycles
_C.TRAIN.NUM_CYCLES = 1
# Number of epoches for each cycle. Length of epoches list must equals number of cycle
_C.TRAIN.EPOCHES = [50]
# Training batchsize
_C.TRAIN.BATCH_SIZE = 32



# ----------------------------------------
# Solver config
# ----------------------------------------
_C.SOLVER = CN()

# Solver algorithm
_C.SOLVER.OPTIMIZER = "adamw"
_C.SOLVER.ADAM_EPS = 1e-8
# Solver scheduler (constant / step / cyclical)
_C.SOLVER.SCHEDULER = "cyclical"
# Warmup length. Set 0 if do not want to use
_C.SOLVER.WARMUP_LENGTH = 0
# Use gradient accumulation. If not used, step equals 1
_C.SOLVER.GD_STEPS = 1
# Starting learning rate (after warmup, if used)
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.MIN_LR=0.
# Weight decay coeffs
_C.SOLVER.WEIGHT_DECAY = 1e-2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# ----------------------------------------
# Loss function config
# ----------------------------------------
_C.LOSS = CN()

# Loss function (ce / focal / dice)
_C.LOSS.NAME = "ce"
_C.LOSS.MSE_CE_WEIGHTS = [1, 1]

# ----------------------------------------
#  Metric config
# ----------------------------------------
_C.METRIC=CN()
_C.METRIC.NAME='dice'
# ----------------------------------------
# Model config
# ----------------------------------------
_C.MODEL = CN()

# Classification model arch
_C.MODEL.NAME = "resnet50"
# Load ImageNet pretrained weights
_C.MODEL.PRETRAINED = True
# Classification head
_C.MODEL.CLS_HEAD = 'linear'
# Number of classification class
_C.MODEL.ISUP_NUM_CLASSES = 6
_C.MODEL.G_ONE_NUM_CLASSES = 6
_C.MODEL.G_TWO_NUM_CLASSES = 6
_C.MODEL.POOL = "adaptive_pooling"
_C.MODEL.ORDINAL = CN({"ENABLED": False})
_C.MODEL.DROPOUT = 0.
_C.MODEL.DROPPATH = 0.
_C.MODEL.CHANNEL_MULTIPLIER = 1.
_C.MODEL.NUM_CLASSES=2
def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
