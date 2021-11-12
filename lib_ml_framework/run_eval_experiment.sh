export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#These args are essential and standard
NOW=$(date +'%Y_%m_%d__%H_%M_%S_%Z')

VERSION=1

source .env
EXPERIMENT_NAME=$EXPERIMENT_NAME
DATASET_DIR=$EVAL_DATASET_DIR_SMALL #If training, this is the training directory. If evaluating, this is the directory of files over which to run evaluation.
CONFIG=$EVAL_CONFIG #If training, make sure this matches the data loader you're using for training ETL. If evaluating, make sure this matches the data loader you're using for evaluation ETL.
OUTPUT_DIR=$OUTPUT_DIR # Used as output dir in training mode and as a model directory in eval mode
OUTPUT_KEY=$OUTPUT_KEY # Used as a column name or dictionary key to store predicted value in a dataset or dictionary

python3 -m src.main \
    --experiment_name $EXPERIMENT_NAME \
    --version $VERSION \
    --output_dir $OUTPUT_DIR \
    --time $NOW \
    --config $CONFIG \
    --data_dir $DATASET_DIR \
    --evaluate \
    --output_key $OUTPUT_KEY