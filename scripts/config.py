import torch

# Configure the training process
LEARNING_RATE = 0.000598419915061102
WEIGHT_DECAY = 1e-06
DROPOUT = 0.1994664332549059
BATCH_SIZE = 512

# Image format
NUM_ROW = 30
NUM_COLUMN = 30
NUM_EPOCHS = 200    # number of epochs to train for
FILE_EXTENSION = ".png"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
print('Running on {}'.format(DEVICE))

# Dataset directory: images and output
OUTPUT_FILE = '../data/outputs/outputs.txt'
IMG_DIR = '../data'

# Dataset split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NUM_WORKERS = 0

# whether to visualize images after creating the data loaders to show the augmentation and class names applied on the img
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45