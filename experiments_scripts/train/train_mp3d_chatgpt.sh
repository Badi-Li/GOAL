export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset mp3d \
               --llm chatgpt \
               --exp_name mp3d_chatgpt \
               --log_dir ./logs/train \
               --epochs 25