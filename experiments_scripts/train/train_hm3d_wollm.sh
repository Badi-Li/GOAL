export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --dataset hm3d \
               --prior \
               --exp_name hm3d_wollm \
               --log_dir ./logs/train \
               --epochs 25