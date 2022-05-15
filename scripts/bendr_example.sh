cd ..
python main.py -dataset Example -multi_thread False -lr 1e-6 -batch_size 8 -num_epochs 50 -strategy PCVOQ -save_dir rgnn -normalized True -mask False -l2_reg 0 -model BENDR
