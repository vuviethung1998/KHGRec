# python main.py --model=DHCF --dataset=lastfm --lrate=0.0001 --n_layers=1 --p=0.2 --drop_rate=0.1 --reg=0.01 --weight_decay=1e-6 --early_stopping_steps=40  --max_epoch=10  
# python main.py --model=DHCF --dataset=ml-1m --lrate=0.0001 --n_layers=1 --p=0.2 --drop_rate=0.1 --reg=0.01 --weight_decay=1e-6 --early_stopping_steps=40  --max_epoch=10  
# python main.py --model=HCCF --dataset=lastfm --lrate=0.001 --p=0.3 --n_layers=2  --drop_rate=0.3 --reg=0.01 --weight_decay=5e-6 --temp=0.1 --cl_rate=5e-5 --early_stopping_steps=50   --max_epoch=10  
# python main.py --model=HCCF --dataset=ml-1m --lrate=0.001 --p=0.3 --n_layers=2  --drop_rate=0.3 --reg=0.01 --weight_decay=5e-6 --temp=0.1 --cl_rate=5e-5 --early_stopping_steps=50   --max_epoch=10  
python main.py --model=KGAT --dataset=lastfm --lrate=0.005 --reg=0.1 --weight_decay=1e-6 --p=0.3 --drop_rate=0.2 --n_layers=2 --early_stopping_steps=20 --gpu_id=1 --max_epoch=200   --max_epoch=10  
python main.py --model=KGAT --dataset=ml-1m --lrate=0.005 --reg=0.1 --weight_decay=1e-6 --p=0.3 --drop_rate=0.2 --n_layers=2 --early_stopping_steps=20 --gpu_id=1 --max_epoch=200   --max_epoch=10  
# python main.py --model=LightGCN --dataset=lastfm --lrate=0.0005 --reg=0.00001 --weight_decay=5e-6 --p=0.3 --drop_rate=0.2 --n_layers=2 --early_stopping_steps=20 --gpu_id=1 --max_epoch=10
# python main.py --model=LightGCN --dataset=ml-1m --lrate=0.001 --reg=0.01 --weight_decay=5e-6 --p=0.3 --drop_rate=0.2 --n_layers=2 --early_stopping_steps=20 --gpu_id=1 --max_epoch=10
# python main.py --model=SGL --dataset=lastfm --lrate=0.005 --n_layers=2 --p=0.1 --drop_rate=0.5 --reg=0.0001 --weight_decay=1e-5 --cl_rate=1e-5 --temp=0.1 --early_stopping_steps=30  --max_epoch=10  
# python main.py --model=SGL --dataset=ml-1m --lrate=0.01 --n_layers=2 --p=0.1 --drop_rate=0.5 --reg=0.0001 --weight_decay=1e-5 --cl_rate=1e-5 --temp=0.1 --early_stopping_steps=30  --max_epoch=10  
# python main.py --model=SHT --dataset=lastfm --lrate=0.005 --n_layers=2 --p=0.3 --drop_rate=0.3 --reg=1e-5 --weight_decay=1e-3 --early_stopping_steps=50  --max_epoch=10  
# python main.py --model=SHT --dataset=ml-1m --lrate=0.005 --n_layers=2 --p=0.3 --drop_rate=0.3 --reg=1e-5 --weight_decay=1e-3 --early_stopping_steps=50   --max_epoch=10  
