cd ..
python -u main.py -dataset AMIGOS -mode unsupervised -save_dir unamigos_lr1e-3b32p3t32 -pos_encoding polar -polar_len 3 -multi_thread True -lr 1e-3 -batch_size 32 -base_path DE4T32/ -strategy PALL
