cd ..
python -u main.py -dataset Example -mode unsupervised -save_dir example_bta_un -pos_encoding polar -polar_len 3 -multi_thread True -lr 1e-3 -batch_size 32 -strategy PCVOQ -num_epochs 1
python main.py -dataset Example -pos_encoding polar -polar_len 3 -multi_thread True -lr 5e-2 -batch_size 8 -num_epochs 1 -strategy PCVOQ -save_dir example_bta_su -normalized True -load_unsupervised_model example_bta_un -d_model 8 -n_heads 8 -mask False -l2_reg 0
