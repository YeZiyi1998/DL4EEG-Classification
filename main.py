from dataloader import MyDataloader, MyDataset, MaskDataset
from torch.utils.data import DataLoader
import torch
import argparse
from model.eegnet import EEGNet
from model.bta import UnsupervisedBTA, SupervisedBTA, get_pos_encoder, BTANet
from model.bendr_model import BENDRClassification
from running import SupervisedRunner, UnsupervisedRunner
import torch.nn as nn
import os
import numpy as np
from model.dgcnn import DGCNN
from model.rgnn import RGNN
from model.het_model import Het
import time
from loss import get_loss_module
import random
from config import dataset2strategy, dataset2path, dataset2strategy2uid
from utils import get_dataset_dict, MaskedMSELoss
import json
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def model_init(args, valid_id, data_class, model_f):
    train, valid = data_class.load_data(valid_id, args.strategy, normalized = args.normalized,)
    dataset_dict = get_dataset_dict(args.dataset, args.model)
    if args.mode == 'supervised':
        train_dataset = MyDataset(train, torch.device(f'cuda:{args.cuda}'), args)
        valid_dataset = MyDataset(valid, torch.device(f'cuda:{args.cuda}'), args)
    else:
        train_dataset = MaskDataset(train, torch.device(f'cuda:{args.cuda}'))
        valid_dataset = MaskDataset(valid, torch.device(f'cuda:{args.cuda}'))
    loss_module = nn.CrossEntropyLoss()
    # temporal feature based model
    if args.model == 'EEGNet' or args.model == 'BENDR':
        model = model_f(args, input_dim = dataset_dict['temp_len'], num_nodes = dataset_dict['max_len'], device= torch.device(f'cuda:{args.cuda}'))
    # temporal&frequency feature based model
    elif args.model == 'Het':
        model = model_f(args, input_dim = dataset_dict['freq_len'], num_nodes = dataset_dict['max_len'], device= torch.device(f'cuda:{args.cuda}'), input_dim2 = dataset_dict['temp_len'])
    # model with subtask
    elif args.model == 'BTA' and args.mode == 'unsupervised':
        pos_encoder = get_pos_encoder(args.pos_encoding)(args.d_model, dropout = 0.1 * (1.0 - 0), max_len = dataset_dict['max_len'], args = args)
        model1 = model_f(feat_dim = dataset_dict['freq_len'], max_len = dataset_dict['max_len'], d_model = args.d_model, n_heads = args.n_heads, num_layers = args.num_layers, dim_feedforward = 128, pos_encoding=pos_encoder, args = args)
        model2 = model_f(feat_dim = dataset_dict['temp_len'], max_len = dataset_dict['max_len'], d_model = args.d_model, n_heads = args.n_heads, num_layers = args.num_layers, dim_feedforward = 128, pos_encoding=pos_encoder, args = args)
        model = UnsupervisedBTA(model1, model2)
        loss_module = MaskedMSELoss()
        return train_dataset, valid_dataset, model, loss_module, UnsupervisedRunner
    # model with subtask
    elif args.model == 'BTA' and args.mode == 'supervised':
        pos_encoder = get_pos_encoder(args.pos_encoding)(args.d_model, dropout = 0.1 * (1.0 - 0), max_len = dataset_dict['max_len'], args = args)
        model1 = model_f(feat_dim = dataset_dict['freq_len'], max_len = dataset_dict['max_len'], d_model = args.d_model, n_heads = args.n_heads, num_layers = args.num_layers, dim_feedforward = 128, pos_encoding=pos_encoder, args = args)
        model2 = model_f(feat_dim = dataset_dict['temp_len'], max_len = dataset_dict['max_len'], d_model = args.d_model, n_heads = args.n_heads, num_layers = args.num_layers, dim_feedforward = 128, pos_encoding=pos_encoder, args = args)
        model = SupervisedBTA(model1, model2, max_len = dataset_dict['max_len'], d_model = args.d_model, num_classes = 2, mask = args.mask)
    # frequency feature based model
    else:
        model = model_f(args, input_dim = dataset_dict['freq_len'], num_nodes = dataset_dict['max_len'], device= torch.device(f'cuda:{args.cuda}')) 
    return train_dataset, valid_dataset, model, loss_module, SupervisedRunner 

def main(args, model_f):
    # load data
    data_class = MyDataloader(args)
    best_metric_list = []
    best_epoch = []
    for valid_id in range(args.start_uid, args.end_uid):
        train_dataset, valid_dataset, model, loss_module, my_runner = model_init(args, valid_id, data_class, model_f)
        
        # create training and validation dataset
        train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle=True,)
        valid_loader = DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle=False,)
        
        # load existing model
        if args.load_unsupervised_model != 'False':
            u_name = valid_id - valid_id % 10
            global_dic = {}
            for i in range(u_name, u_name + 10):
                unsupervices_model_state_dict = torch.load(f'models/{args.load_unsupervised_model}/{i}.dic.pkl', map_location='cpu')['state_dict']
                for key in unsupervices_model_state_dict.keys():
                    if key not in global_dic.keys():
                        global_dic[key] = [unsupervices_model_state_dict[key].cpu().numpy().tolist()]
                    else:
                        global_dic[key].append(unsupervices_model_state_dict[key].cpu().numpy().tolist())
            model_dic = model.state_dict()
            for key in global_dic.keys():
                if 'pos_enc' in key and key in model_dic.keys():
                    model_dic[key] = torch.mean(torch.tensor(global_dic[key], dtype=torch.float), dim=0,)
            model.load_state_dict(model_dic)
        
        model.to(torch.device(f'cuda:{args.cuda}'))
        
        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, )
        
        # create runner
        runner = my_runner(model, train_loader, valid_loader, torch.device(f'cuda:{args.cuda}'), loss_module, optimizer, print_interval=30, batch_size=args.batch_size, l2_reg = args.l2_reg)
        
        # initilize best_metric
        if args.mode == 'unsupervised':
            best_metric = 1e23
        elif args.mode == 'supervised':
            best_metric = 0
        
        # initlize recoders
        best_predictions = []
        early_stop = 20
        early_stop_num = 0
        best_epoch.append(-1)
        auc_list = []

        for i in range(args.num_epochs):
            time_start=time.time()
            
            # start training
            if args.evaluate != True:
                epoch_metric = runner.train_epoch(i)
            else:
                epoch_metric = None
            auc, acc, total_predictions, total_Y, loss = runner.evaluate()
            auc_list.append(auc)
            time_end = time.time()
            time_cost = time_end - time_start

            # logging
            print('thread_id {:.1f} Epoch {:.1f} training loss: {:.4f} r loss: {:.4f} validation metric: {:.3f}, loss: {:.4f}, time cost: {:.3f}'.format(args.thread_id, epoch_metric['epoch'], epoch_metric['loss'], epoch_metric['rl'], auc, loss, time_cost))
            
            # save model and results
            if args.mode == 'unsupervised' and auc < best_metric or args.mode == 'supervised' and auc > best_metric:
                torch.save({'state_dict': model.state_dict()}, os.path.join('models/' + args.save_dir, f'{valid_id}.dic.pkl'))
                best_metric = auc
                best_predictions = total_predictions
                early_stop_num = 0
                best_epoch[-1] = i
            else:
                early_stop_num += 1

            # early stop
            if early_stop_num > early_stop:
                break
            
            # avoid learning rate too large
            if early_stop_num > 10 and max(auc_list[-5:]) < 0.55:
                for name, param in model.named_parameters():
                    torch.nn.init.normal_(param, mean=0, std=0.01)
                optimizer = torch.optim.Adam(model.parameters(), lr = args.lr * 0.2)

        with open(os.path.join('results/' + args.save_dir, f'{valid_id}.txt'), 'w') as fw:
            fw.write(str(best_predictions) + '\n' + str(total_Y) + '\n' + str(best_metric))
        best_metric_list.append(best_metric)
    print(f'valid_id: {valid_id} mean metric:', np.mean(best_metric_list), 'best epoch list', best_epoch)
    return best_metric_list

def print_error(value):
    print("multi_thread_error:", value)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', type=str, help='Training strategy. PCVOQ: task dependent training for each pariticipant; PALL: random shuffled training for each pariticipant', default = 'PCVOQ', choices = ['PALL','PCVOQ'])
    parser.add_argument('-cuda', type=int, default = 0, required=False, help='available cuda for single thread training')
    parser.add_argument('-cuda_list', type=str, default = '[0,1,2,3,4]', required=False, help='available cuda list for multi thread training')
    parser.add_argument('-batch_size', type=int,default = 8, required=False)
    parser.add_argument('-lr', type=float, default = 5e-3, required=False)
    parser.add_argument('-save_dir', type=str,default = 'tmp', required=False)
    parser.add_argument('-num_epochs', type=int,default = 50, required=False)
    parser.add_argument('-model', type=str,default = 'BTA', required=False, choices = ['DGCNN', 'EEGNet', 'RGNN', 'Het', 'BENDR','BTA'])
    parser.add_argument('-mode', type=str,default = 'supervised', required=False, choices = ['supervised', 'unsupervised'])
    parser.add_argument('-load_unsupervised_model', type=str, default = 'False', required=False, help='unsupervised model path, False if no unsupervised model loaded for supervised task')
    parser.add_argument('-normalized', type=str, default = 'True', required=False)
    parser.add_argument('-mask', type=str,default = 'False', required=False, choices = ['frequency','temporal','False'], help='mask temporal or frequency features.',) 
    parser.add_argument('-base_path', type=str,default = 'DE5T62/', required=False)
    parser.add_argument('-num_layers', type=int,default = 1, required=False)
    parser.add_argument('-d_model', type=int, default = 8, required=False)
    parser.add_argument('-n_heads', type=int, default = 8, required=False)
    parser.add_argument('-evaluate', type=str,default = 'False', required=False, help='evaluate mode or not. if True, no training process is included')
    parser.add_argument('-start_uid', type=int,default = -1, required=False, help='training file start id')
    parser.add_argument('-end_uid', type=int,default = -1, required=False, help='training file end id')
    parser.add_argument('-multi_thread', type=str, default = 'False', required=False, help='multi thread training')
    parser.add_argument('-pos_encoding', type=str, default = 'polar', required=False)
    parser.add_argument('-polar_len', type=int, default = 3, required=False, help='number of centralities')
    parser.add_argument('-dataset', type=str, default = 'AMIGOS', choices = ['Example', 'AMIGOS', 'Search-Brainwave'], required=False)
    parser.add_argument('-split_mode', type=str, default = '1_5', choices = ['1_5', '123_45', '1234_5', '1_2345'], required=False, help='split mode to split the search-brainwave, default is 1_5, the same as the original paper')
    parser.add_argument('-l2_reg', type=float, default = 0, required=False)
    parser.add_argument('-l1_reg', type=float, default = 0, required=False)
    
    args = parser.parse_args() 
    args.normalized = False if args.normalized == 'False' else True
    args.evaluate = False if args.evaluate == 'False' else True
    args.multi_thread = False if args.multi_thread == 'False' else True
    args.cuda_list = json.loads(args.cuda_list)

    if args.multi_thread:
        torch.multiprocessing.set_start_method('spawn')

    # check args
    if args.strategy not in dataset2strategy[args.dataset]:
        print(f'{args.dataset} doesn\'t support strategy {args.strategy}')
        exit()
    
    if args.start_uid == -1:
        args.start_uid = dataset2strategy2uid[args.dataset][args.strategy][0]
        args.end_uid = dataset2strategy2uid[args.dataset][args.strategy][1]
    
    if args.model == 'EEGNet' or args.model == 'BENDR':
        args.base_path = 'DE5T128/' if args.dataset == 'AMIGOS' else 'DE5T1251/'
        if args.model == 'EEGNet':
            model_f = EEGNet
        else:
            model_f = BENDRClassification
    else:
        if args.model == 'DGCNN':
            model_f = DGCNN
        elif args.model == 'RGNN':
            model_f = RGNN
        elif args.model == 'Het':
            model_f = Het
        elif args.model == 'BTA':
            model_f = BTANet
    args.base_path = dataset2path[args.dataset] + args.base_path
    return args, model_f


if __name__ == '__main__':
    # get args
    args, model_f = init()
     
    # create output data directory
    if os.path.exists('models/' + args.save_dir) == False:
        os.mkdir('models/' + args.save_dir)
        os.mkdir('results/' + args.save_dir)
    json.dump(vars(args), open(f'results/{args.save_dir}/information_{time.time()}.json', 'w'))

    if args.multi_thread:
        from multiprocessing.pool import Pool
        import copy
        all_metric_list = []
        k_max = int(500 / len(args.cuda_list)) + 1
        for k in range(0, k_max):
            pool = Pool(len(args.cuda_list) + 1)
            item_list = []
            for i in range(k * len(args.cuda_list), k * len(args.cuda_list) + len(args.cuda_list)):
                if i < args.start_uid or i >= args.end_uid:
                    continue
                args_new = copy.deepcopy(args)
                args_new.cuda = args.cuda_list[(i - k * len(args.cuda_list)) % len(args.cuda_list)]
                args_new.thread_id = (i - k * len(args.cuda_list)) % len(args.cuda_list)
                args_new.start_uid = i
                args_new.end_uid = i + 1
                item_list.append(pool.apply_async(main, args=(args_new, model_f), error_callback=print_error))
            pool.close()
            pool.join()
            for item in item_list:
                all_metric_list.append(item.get())
    else:
        args.thread_id = 0
        all_metric_list = main(args, model_f)
    print("np.mean(all_metric_list)", np.mean(all_metric_list))
