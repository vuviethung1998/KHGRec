CUDA_VISIBLE_DEVICES=0 python main.py --model=HGNNAblation --dataset=lastfm --mode=wossl --lrate=0.0001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=0.0001 --temp=0.2 --reg=0.1 --early_stopping_steps=50