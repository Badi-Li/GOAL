export CUDA_VISIBLE_DEVICES=0,1,6,7
python main.py --dataset mp3d \
               --llm chatgpt \
               --exp_name mp3d_chatgpt \
               --log_dir ./logs/train \
               --epochs 25 \
               --resume /home/badi/GOAL/logs/train/mp3d_chatgpt/goal-epoch10.pth 