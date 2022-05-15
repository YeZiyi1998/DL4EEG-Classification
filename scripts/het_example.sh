cd ..
python main.py -dataset Example -multi_thread False -lr 5e-3 -batch_size 8 -num_epochs 50 -strategy PCVOQ -save_dir rgnn -normalized True -mask False -l2_reg 0 -model Het -start_uid 0 -end_uid 1
