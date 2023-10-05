<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python main.py --model=HGNN --dataset=ml-1m --n_layers=3 --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.3 --p=0.3 --cl_rate=0.01 --temp=1 --reg=0.1 --early_stopping_steps=50
=======
CUDA_VISIBLE_DEVICES=0 python main.py --model=HGNN --dataset=ml-1m --n_layers=1 --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=0.0001 --temp=0.2 --reg=0.1 --early_stopping_steps=50
CUDA_VISIBLE_DEVICES=0 python main.py --model=HGNN --dataset=ml-1m --n_layers=2 --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=0.0001 --temp=0.2 --reg=0.1 --early_stopping_steps=50
CUDA_VISIBLE_DEVICES=0 python main.py --model=HGNN --dataset=ml-1m --n_layers=3 --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=0.0001 --temp=0.2 --reg=0.1 --early_stopping_steps=50
CUDA_VISIBLE_DEVICES=0 python main.py --model=HGNN --dataset=ml-1m --n_layers=4 --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=0.0001 --temp=0.2 --reg=0.1 --early_stopping_steps=50
>>>>>>> 33c613c75d424dde82363171af74047fdd185d47
