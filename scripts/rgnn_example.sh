cd ..
python main.py -dataset Example -multi_thread False -lr 5e-3 -batch_size 8 -num_epochs 50 -strategy PCVOQ -save_dir rgnn -normalized True -mask False -l2_reg 1e-3 -model RGNN -cuda 8
