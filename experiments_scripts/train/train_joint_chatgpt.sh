export CUDA_VISIBLE_DEVICES=2,3,4,6
python main.py --dataset joint \
               --llm chatgpt \
               --exp_name joint_chatgpt \
               --log_dir ./logs/train \
               --epochs 15 \
               --save_floor 10