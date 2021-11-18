export CUDA_VISIBLE_DEVICES=0,1

#These args are essential and standard
NOW=$(date +'%Y_%m_%d__%H_%M_%S_%Z')

VERSION=1

EXPERIMENT_NAME='tweet_experiment'
DATASET_DIR='resources/datasets/spam_predict' #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
CONFIG='src/config/nlp/config_sms_spam.json' #If training, make sure this matches the data loader you're using for training ETL. If evaluating, make sure this matches the data loader you're using for evaluation ETL.
OUTPUT_DIR='outputs/spam' # Used as output dir in training mode and as a model directory in eval mode
OUTPUT_KEY='prediction' # Used as a column name or dictionary key to store predicted value in a dataset or dictionary

python3 -m src.main \
    --experiment_name $EXPERIMENT_NAME \
    --version $VERSION \
    --output_dir $OUTPUT_DIR \
    --time $NOW \
    --config $CONFIG \
    --data_dir $DATASET_DIR \
    --evaluate \
    --output_key $OUTPUT_KEY