export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset hm3d \
               --llm chatgpt \
               --exp_name hm3d_chatgpt \
               --log_dir ./logs/train \
               --epochs 25