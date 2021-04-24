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
_C.DATA.NUM_FOLD=5
_C.DATA.FOLD=0
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
# For ImageFolder dataset style
# TODO #
# Random Resized Crop when training
_C.DATA.CROP = CN({"ENABLED": False})
_C.DATA.CROP.SCALE = (0.7, 1.0)
_C.DATA.CROP.RATIO = (0.75, 1.333333)
_C.DATA.CROP.INTERPOLATION = 2
# Apply Z-Norm for dataloader
_C.DATA.NORMALIZE = CN({"ENABLED": False})
_C.DATA.NORMALIZE.MEAN_R = []
_C.DATA.NORMALIZE.STD_R = []
_C.DATA.NORMALIZE.MEAN_K = []
_C.DATA.NORMALIZE.STD_K = []
# Dataset augmentations style (albumentations / randaug / augmix)
_C.DATA.AUGMENT = ""
# For randaug augmentation. For augmix or albumentations augmentation, refer to those other section
_C.DATA.RANDAUG = CN()
# Choose RandAugment config of augmentations
_C.DATA.RANDAUG.CONFIG = 1
# Number of augmentations picked for each iterations. Default is 2
_C.DATA.RANDAUG.N = 2
# Amptitude of augmentation transform (0 < M < 30). Default is 27
_C.DATA.RANDAUG.M = 27
_C.DATA.RANDAUG.RANDOM_MAGNITUDE = False
# For augmix augmentaion
_C.DATA.AUGMIX = CN()
_C.DATA.AUGMIX.ALPHA = 1.
_C.DATA.AUGMIX.BETA = 1.
# For albumentations augmentation
# TODO #
# Cutmix data transformation for training
_C.DATA.CUTMIX = CN({"ENABLED": False})
_C.DATA.CUTMIX.ALPHA = 1.0
# Mixup data transformation for training
_C.DATA.MIXUP = CN({"ENABLED": False})
_C.DATA.MIXUP.ALPHA = 1.0
# Gridmask data transformation for training
_C.DATA.GRIDMASK = CN({"ENABLED": False})
_C.DATA.BALANCED_SAMPLING = CN({"ENABLED": False})
_C.DATA.INVERT = False


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
# Inference config
# ----------------------------------------
_C.INFER = CN()
_C.INFER.TTA = CN({"ENABLED": False})
# Horizontal flip TTA
_C.INFER.TTA.HFLIP = False
# Vertical flip TTA
_C.INFER.TTA.VFLIP = False
# TenCrop TTA
_C.INFER.TTA.CROP = False
_C.INFER.SAVE_PREDICTION = CN({"ENABLED": False})


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
# Stochastic weights averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
# SWA starting point (epoches, iterations)
_C.SOLVER.SWA.START_EPOCH = 0
# SWA frequency (epoches, iterations)
_C.SOLVER.SWA.FREQ = 5
# SWA decay coeff
_C.SOLVER.SWA.DECAY = 0.999
_C.SOLVER.SWA.MEAN_TEACHER_MSE = False
_C.SOLVER.SWA.MT_START_EPOCH = 0

# ----------------------------------------
# Loss function config
# ----------------------------------------
_C.LOSS = CN()

# Loss function (ce / focal / dice)
_C.LOSS.NAME = "ce"
_C.LOSS.MSE_CE_WEIGHTS = [1, 1]


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
_C.MODEL.NESTED_UNET=CN({"ENABLED": False})
_C.MODEL.NESTED_UNET.DEEP_SUPERVISION= True 
_C.MODEL.NESTED_UNET.DIMS= []
def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
