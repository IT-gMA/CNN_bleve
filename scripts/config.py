import torch

# Random shuffle of dataset
SEED_RANGE = 4

# Configure the training process
LEARNING_RATE = 1e-3 #0.0033808570389529695
MIN_LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-06
USE_DROP_OUT = False
DROPOUT = 0.2477008537587833
VARRY_LR = False
SCHEDULED = True
MSE_REDUCTION = "mean"
PATIENCE = 16
BATCH_SIZE = 256
VAL_BATCHSIZE = 256
PRINT_TRAIN = False
PRINT_VAL = False
PRINT_TEST = False

# Image format
RESCALE = True
NUM_ROW = 240
NUM_COLUMN = 240
NUM_EPOCHS = 1000    # number of epochs to train for
FILE_EXTENSION = ".png"
TRAIN_AUG = False

# Model config
RESUME = False
MODEL_NAME = "resnet18"
USE_PRETRAIN = True
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
EMPTY_CUDA_CACHE = False
print('Running on {}'.format(DEVICE))
VALIDATION_EPOCH = 5

# Dataset directory: images and output
OUTPUT_FILE = '../dataset/data/outputs/outputs.txt'
IMG_DIR = '../dataset/data'
SAVE_IMG_DIR = '../temp_dataset/DI_bleve_40x40_large_frame'
SAVE_OUTPUT_DIR = '../temp_dataset/DI_bleve_40x40_large_frame/outputs/outputs.txt'

# Dataset split
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
TRAIN_RATIO = 0.74
VAL_RATIO = 0.13
TEST_RATIO = 0.13

ORDER = True
ORDER_METHOD = 1
NUM_WORKERS = 0

# location to save model and plots
SAVED_MODEL_DIR = "../saved_models/resnet18v15"
SAVED_MODEL_NAME = "resnet18v15"
SAVED_MODEL_FORMAT = ".pt"

SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

# store model's run history
LOG_DIR = "../running_logs/resnet18v15"


# load model from
LOAD_MODEL_LOCATION = "../saved_models/resnet18v10/seed_3/resnet18v10_801_model.pt"

# wandb running config
WANDB_PROJECT_NAME = "CNN_bleve_resnet18v15"


# the size of tensor arrays being displayed
NP_FULL_SIZE = False

# nth epoch at which a checkpoint is saved
SAVED_EPOCH = 200
SAVED_AFTER = 499

# Progress bar
PROGRESS_SLEEP_TIME = 0.00002