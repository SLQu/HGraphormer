import torch
import numpy as np
import os,sys
import config,utils
import path
import time
from datetime import datetime
from logger import log
import datasets
from transformer import HGraphormer


if __name__ == '__main__':
    args = config.parse()
    args.cmd_input = 'python ' + ' '.join(sys.argv) + '\n'

    # configure output directory
    root_dir = str(os.getcwd())
    out_dir = path.Path(root_dir + os.sep + args.out_dir + os.sep + f'{args.data}_{args.dataset}')

    ## configure logger, creat result folder
    start_time = time.strftime("%Y%m%d%H%M%S")+'_'+str(datetime.now().time()).split('.')[-1]
    args.start_time = start_time
    log_name =   f'{args.model_name}_{start_time}_logging.log'
    _path_1_ = f'{out_dir}{os.sep}'
    _path_2_ = f'store_{args.model_name}_{start_time}{os.sep}'
    utils.ckeckMakeDir(_path_1_+_path_2_)
    log = log(_path_1_+log_name,_path_1_+_path_2_+log_name)
    args.log = log
    utils.storeFile(_path_1_ + _path_2_)

    # set seed and get gpu information
    device = utils.setEnv(args)
    log.info('GPU is '+str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        utils.gpuInfo(log)

    # load data
    X, Y, G, Source2Id = datasets.fetch_data(args)
    # dataset,Source2Id = datasets.loaddata(args)
    args.node_num = Y.shape[0]
    args.edge_num = len(G)
    args.nfeat = X.shape[1]
    args.nclass = torch.max(Y).item()+1
    
    args.snn = args.snn if args.node_num > args.snn  else args.node_num
    args.maxlen = args.snn 

    test_accs,best_val_accs, best_test_accs = [], [], []    
    log.info(args)

    for run in range(1, args.nruns+1):
        
        # gpu, seed
        log.info('GPU is '+str(torch.cuda.is_available()))
        log.info('Total Epochs: {args.epochs}')
        
        # load data
        args.split = run 
        train_idx, test_idx = datasets.lable(Source2Id,args)
        
        model = HGraphormer(args).to(device)
        
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=0.01, weight_decay=5e-4)
        log.info('Total Epochs: {args.epochs}')
        log.info(model)
        log.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )
        tic_run = time.time()
        best_test_acc, test_acc = 0, 0,    
        Y = Y.to(device)
        
        for epoch in range(args.epochs):
            
            sub_H = torch.tensor(datasets.getNumpyH(G,args.node_num,args.edge_num),dtype=torch.float32)
            sub_full_idx = torch.LongTensor(list(range(args.node_num)))   
            sub_eval_idx_ful = torch.LongTensor(list(train_idx))  
            sub_eval_idx_sub = torch.LongTensor(list(train_idx))  
            node2edge,edge2node = None, None
            train_subgraphs=[[sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,node2edge,edge2node]]

            # train
            tic_epoch = time.time()
            model.train()

            train_acc=[]
            count = 0
            
            for sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,node2edge,edge2node in train_subgraphs:
                _sub_X_ = X[sub_full_idx]
                _sub_X_ = _sub_X_.to(device)
                sub_full_idx = sub_full_idx.to(device)
                
                sub_H,sub_eval_idx_ful,sub_eval_idx_sub = sub_H.to(device),sub_eval_idx_ful.to(device),sub_eval_idx_sub.to(device)

                optimizer.zero_grad()
                Z = model(device,_sub_X_,sub_H,sub_full_idx,node2edge,edge2node)
                loss = torch.nn.functional.nll_loss(Z[sub_eval_idx_sub], Y[sub_eval_idx_ful])
                train_acc.append(Z[sub_eval_idx_sub].argmax(1).eq(Y[sub_eval_idx_ful]).float())
                del sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,Z
                
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                
                if len(train_acc)==0:
                    train_acc = 0
                else:
                    train_acc = torch.cat(train_acc).mean().item()
                    
                train_time = time.time() - tic_epoch 
        ############################################################################
        
        # eval
        model.eval()
        test_acc=[]

        sub_H = torch.tensor(datasets.getNumpyH(G,args.node_num,args.edge_num),dtype=torch.float32)
        sub_full_idx = torch.LongTensor(list(range(args.node_num)))   
        sub_eval_idx_ful = torch.LongTensor(list(test_idx))  
        sub_eval_idx_sub = torch.LongTensor(list(test_idx))  
        node2edge,edge2node = None, None
        test_subgraphs=[[sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,node2edge,edge2node]]
                
        for sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,node2edge,edge2node in test_subgraphs:

            _sub_X_ = X[sub_full_idx]
            _sub_X_ = _sub_X_.to(device)
            sub_full_idx = sub_full_idx.to(device)
            sub_H,sub_eval_idx_ful,sub_eval_idx_sub = sub_H.to(device),sub_eval_idx_ful.to(device),sub_eval_idx_sub.to(device)
            
            Z,_ = model(device,_sub_X_,sub_H,sub_full_idx,node2edge,edge2node)
            test_acc.append(Z[sub_eval_idx_sub].argmax(1).eq(Y[sub_eval_idx_ful]).float())
            del sub_H,sub_full_idx,sub_eval_idx_ful,sub_eval_idx_sub,Z

            torch.cuda.empty_cache()
        if len(test_acc)==0:
            test_acc = 0
        else:
            test_acc = torch.cat(test_acc).mean().item()

        # log acc
        best_test_acc = max(best_test_acc, test_acc)
        log.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.4f} | test acc:{test_acc:.4f} | time:{train_time*1000:.1f}ms')
    
    
    
    
    
    
    print('---end---')
