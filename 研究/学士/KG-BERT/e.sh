

python3 mlm_tail.py \
    --do_train \
    --do_eval \
    --data_dir ./data/WN18RR \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --output_dir ./mlm_result/output_tail_3/2\
    --gradient_accumulation_steps 1 \
    --eval_batch_size 128 \

# python3 run_bert_mlm.py \
# python3 run_bert_relation_prediction.py \
# python3 run_bert_link_prediction.py \
# python3 run_bert_entity_prediction.py \
    # --task_name kg \
    # --do_train  \
    # --do_eval \
    # --do_predict \
    # --data_dir ./data/WN18RR \
    # --bert_model bert-base-cased \
    # --max_seq_length 50 \
    # --train_batch_size 32 \
    # --learning_rate 5e-5 \
    # --num_train_epochs 5.0 \
    # --output_dir ./result/output_WN18RR_250_2 \
    # --gradient_accumulation_steps 1 \
    # --eval_batch_size 128 \
