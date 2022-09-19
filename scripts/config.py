import torch

# Random shuffle of dataset
SEED_RANGE = 5

# Configure the training process
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-06
USE_DROP_OUT = False
DROPOUT = 0.1994664332549059
VARRY_LR = False
SCHEDULED = True
MSE_REDUCTION = "mean"
PATIENCE = 16
BATCH_SIZE = 512
VAL_BATCHSIZE = 256
PRINT_TRAIN = False
PRINT_VAL = False
PRINT_TEST = False
USE_PRETRAIN = False

# Image format
RESCALE = True
NUM_ROW = 168
NUM_COLUMN = 168
NUM_EPOCHS = 1800    # number of epochs to train for
FILE_EXTENSION = ".png"

MODEL_NAME = "resnet18"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
EMPTY_CUDA_CACHE = False
print('Running on {}'.format(DEVICE))

# Dataset directory: images and output
OUTPUT_FILE = '../dataset/data/outputs/outputs.txt'
IMG_DIR = '../dataset/data'
SAVE_IMG_DIR = '../dataset_run/bleve_no_errv4'
SAVE_OUTPUT_DIR = '../dataset_run/bleve_no_errv4/outputs/outputs.txt'

# Dataset split
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
TRAIN_RATIO = 0.76
VAL_RATIO = 0.12
TEST_RATIO = 0.12

ORDER = True
ORDER_METHOD = 1
NUM_WORKERS = 0

# location to save model and plots
SAVED_MODEL_DIR = "../saved_models/resnet18v8"
SAVED_MODEL_NAME = "resnet18v8"
SAVED_MODEL_FORMAT = ".pt"

SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

# store model's run history
LOG_DIR = "../running_logs/resnet18v8"


# load model from
LOAD_MODEL_LOCATION = "../saved_models/resnet18v2/resnet18v2best_model.pt"

# wandb running config
WANDB_PROJECT_NAME = "CNN_bleve_resnet18v9"


# the size of tensor arrays being displayed
NP_FULL_SIZE = False

# nth epoch at which a checkpoint is saved
SAVED_EPOCH = 100