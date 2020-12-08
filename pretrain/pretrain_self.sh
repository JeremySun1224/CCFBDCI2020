export TRAIN_FILE=../data/train/all_content.txt
export LTP_RESOURCE=../pyltp/ltp.model
export BERT_RESOURCE=./tokenization_bert.py
export SAVE_PATH=../data/ref.txt


python run_mlm_wwm.py --model_name_or_path hfl/chinese-roberta-wwm-ext --train_file ./pretrain_data/pretrain_train.txt --validation_file ./pretrain_data/pretrain_val.txt --train_ref_file ./pretrain_data/ref_train.txt --validation_ref_file ./pretrain_data/ref_val.txt --do_train --do_eval --output_dir ./output
