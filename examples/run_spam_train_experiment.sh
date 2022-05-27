export CUDA_VISIBLE_DEVICES=0

#These args are essential and standard
NOW=$(date +'%Y_%m_%d__%H_%M_%S_%Z')

VERSION=1

EXPERIMENT_NAME='tweet_experiment'
DATASET_DIR='datasets/spam' #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
CONFIG='config/config_sms_spam.json' #If training, make sure this matches the data loader you're using for training ETL. If evaluating, make sure this matches the data loader you're using for evaluation ETL.
OUTPUT_DIR='outputs/spam' # Used as output dir in training mode and as a model directory in eval mode

python -m main \
    --experiment_name $EXPERIMENT_NAME \
    --version $VERSION \
    --output_dir $OUTPUT_DIR \
    --time $NOW \
    --config $CONFIG \
    --data_dir $DATASET_DIR \
