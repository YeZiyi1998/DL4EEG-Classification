cd ..
python -u main.py -dataset AMIGOS -base_path DE4T32/ -save_dir amigos -mask False -multi_thread True -lr 1e-2 -batch_size 32 -strategy PALL -normalized True -l2_reg 1e-3
