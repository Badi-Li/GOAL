export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset joint \
               --prior \
               --exp_name joint_wollm \
               --log_dir ./logs/train \
               --epochs 25