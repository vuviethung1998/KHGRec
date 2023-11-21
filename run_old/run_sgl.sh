python main.py --model=SGL --dataset=lastfm --lrate=0.005 --n_layers=2 --p=0.1 --drop_rate=0.5 --reg=0.0001 --weight_decay=1e-5 --cl_rate=1e-5 --temp=0.1 --early_stopping_steps=30
python main.py --model=SGL --dataset=ml-1m --lrate=0.01 --n_layers=2 --p=0.1 --drop_rate=0.5 --reg=0.0001 --weight_decay=1e-5 --cl_rate=1e-5 --temp=0.1 --early_stopping_steps=30
