import torch

# Configure the training process
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 0.0005598419915061102
WEIGHT_DECAY = 1e-06
USE_DROP_OUT = True
DROPOUT = 0.1994664332549059
VARRY_LR = True
SCHEDULED = True
MSE_REDUCTION = "mean"
PATIENCE = 16
BATCH_SIZE = 4096
VAL_BATCHSIZE = 512
PRINT_TRAIN = False
PRINT_VAL = False
PRINT_TEST = False

# Image format
RESCALE = True
NUM_ROW = 30
NUM_COLUMN = 30
NUM_EPOCHS = 20000    # number of epochs to train for
FILE_EXTENSION = ".png"

MODEL_NAME = "resnet18"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
EMPTY_CUDA_CACHE = False
print('Running on {}'.format(DEVICE))

# Dataset directory: images and output
OUTPUT_FILE = '../dataset/data/outputs/outputs.txt'
IMG_DIR = '../dataset/data'
SAVE_IMG_DIR = '../dataset_run/bleve_ordered_e1'
SAVE_OUTPUT_DIR = '../dataset_run/bleve_ordered_e1/outputs/outputs.txt'

# Dataset split
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

ORDER = True
ORDER_METHOD = 1
NUM_WORKERS = 0

# location to save model and plots
SAVED_MODEL_DIR = "../saved_models/resnet18v2"
SAVED_MODEL_NAME = "resnet18v2"
SAVED_MODEL_FORMAT = ".pt"

SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

# store model's run history
LOG_DIR = "../running_logs/resnet18v2"


# load model from
LOAD_MODEL_LOCATION = "../saved_models/resnet18/resnet18best_model.pt"

# wandb running config
WANDB_PROJECT_NAME = "CNN_bleve_resnet18v2"


# the size of tensor arrays being displayed
NP_FULL_SIZE = False
