cd ..
python main.py -dataset Search-Brainwave -pos_encoding polar -polar_len 3 -multi_thread True -lr 5e-2 -batch_size 8 -num_epochs 50 -strategy PCVOQ -save_dir search-brainwave -normalized True -load_unsupervised_model unp3lr1e-3b32PCVOQ -d_model 8 -n_heads 8 -mask False -l2_reg 0
