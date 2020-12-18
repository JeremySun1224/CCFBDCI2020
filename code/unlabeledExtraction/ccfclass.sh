
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0




DATASET=all_dataset
LABEL_NAME_FILE=label_names.txt
TRAIN_CORPUS=train.txt
TEST_CORPUS=test.txt
TEST_LABEL_FILE=test_labels.txt
MAX_LEN=300
TRAIN_BATCH=16
ACCUM_STEP=8
EVAL_BATCH=64
GPUS=1
MCP_EPOCH=30
SELF_TRAIN_EPOCH=15
TOP_PRED_NUM=30
CATEGORY_VOCAB_SIZE=50
MATCH_THRESHOLD=10
DIST_PORT=12345

python src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} \
                    --test_file ${TEST_CORPUS} \
                    --test_label_file ${TEST_LABEL_FILE} \
                    --top_pred_num ${TOP_PRED_NUM} \
                    --category_vocab_size ${CATEGORY_VOCAB_SIZE} \
                    --match_threshold ${MATCH_THRESHOLD} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} \
                    --dist_port ${DIST_PORT} \
