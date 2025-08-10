export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset joint \
               --llm chatgpt \
               --exp_name joint_chatgpt \
               --log_dir ./logs/train \
               --epochs 15