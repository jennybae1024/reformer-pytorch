num_layer="12"
d_model="512"
max_len="128"
bucket_length="32"


python train.py \
    --num_layers ${num_layer} \
    --d_model ${d_model} \
    --max_len ${max_len} \
    --bucket_length ${bucket_length} 