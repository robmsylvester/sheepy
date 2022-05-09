export CUDA_VISIBLE_DEVICES=0,1

#These args are essential and standard
NOW=$(date +'%Y_%m_%d__%H_%M_%S_%Z')

VERSION=1

EXPERIMENT_NAME='semeval_experiment'
TRAIN_DATASET_DIR='resources/datasets/semeval/semeval-2017-train.csv' #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
VAL_DATASET_DIR='resources/datasets/semeval/semeval-2017-dev.csv' #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
TEST_DATASET_DIR='resources/datasets/semeval/semeval-2017-test.csv' #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
CONFIG='src/config/examples/config_semeval.json' #If training, make sure this matches the data loader you're using for training ETL. If evaluating, make sure this matches the data loader you're using for evaluation ETL.
OUTPUT_DIR='outputs/semeval' # Used as output dir in training mode and as a model directory in eval mode
OUTPUT_KEY='prediction' # Used as a column name or dictionary key to store predicted value in a dataset or dictionary

python -m src.main \
    --experiment_name $EXPERIMENT_NAME \
    --version $VERSION \
    --output_dir $OUTPUT_DIR \
    --time $NOW \
    --config $CONFIG \
    --train_data_dir $TRAIN_DATASET_DIR \
    --val_data_dir $VAL_DATASET_DIR \
    --test_data_dir $TEST_DATASET_DIR \