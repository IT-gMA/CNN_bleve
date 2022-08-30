import torch

# Configure the training process
LEARNING_RATE = 0.000598419915061102
MIN_LEARNING_RATE = 0.000598419915061102
WEIGHT_DECAY = 1e-06
DROPOUT = 0.1994664332549059
VARRY_LR = True
SCHEDULED = True
MSE_REDUCTION = "mean"
PATIENCE = 16
BATCH_SIZE = 256
VAL_BATCHSIZE = 128
PRINT_TRAIN = False
PRINT_VAL = False
PRINT_TEST = False

# Image format
RESCALE = True
NUM_ROW = 30
NUM_COLUMN = 30
NUM_EPOCHS = 100    # number of epochs to train for
FILE_EXTENSION = ".png"

MODEL_NAME = "resnet50"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
print('Running on {}'.format(DEVICE))

# Dataset directory: images and output
OUTPUT_FILE = '../dataset/data/outputs/outputs.txt'
IMG_DIR = '../dataset/data'
SAVE_IMG_DIR = '../dataset_run/perma_data_full'
SAVE_OUTPUT_DIR = '../dataset_run/perma_data_full/outputs/outputs.txt'

# Dataset split
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

NUM_WORKERS = 0

# location to save model and plots
SAVED_MODEL_DIR = "../saved_models/resnet50"
SAVED_MODEL_NAME = "restnet50_46k"
SAVED_MODEL_FORMAT = ".pt"

SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

# store model's run history
LOG_DIR = "../running_logs/resnet50"


# load model from
LOAD_MODEL_LOCATION = ""


# the size of tensor arrays being displayed
NP_FULL_SIZE = False
