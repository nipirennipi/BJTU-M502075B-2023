# data
FILTER_NUM=3
TITLE_SIZE=20
MAX_HIS_SIZE=50

# model
NEWS_DIM=128
WINDOW_SIZE=3

# train and inference
EPOCHS=2
TRAIN_BATCH_SIZE=64
INFER_BATCH_SIZE=256
LEARNING_RATE=0.0001

python train.py \
    --filter_num $FILTER_NUM \
    --title_size $TITLE_SIZE \
    --max_his_size $MAX_HIS_SIZE \
    --news_dim $NEWS_DIM \
    --window_size $WINDOW_SIZE \
    --epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \

echo -e "\n \033[36m --- Training done, making predictions! --- \033[0m \n"

python predict.py \
    --filter_num $FILTER_NUM \
    --title_size $TITLE_SIZE \
    --max_his_size $MAX_HIS_SIZE \
    --news_dim $NEWS_DIM \
    --window_size $WINDOW_SIZE \
    --infer_batch_size $INFER_BATCH_SIZE \
