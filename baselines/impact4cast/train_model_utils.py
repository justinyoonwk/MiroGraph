import os
import pickle
import gzip
from datetime import datetime, date
import torch
from torch import nn
import torch.nn.functional as F
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from collections import defaultdict,Counter
from itertools import combinations
import pandas as pd
import copy
from features_utils import *
from preprocess_utils import *
from general_utils import * 

#####--------------neural network----------------------------#####
class ff_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ff_network, self).__init__()

        self.semnet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        res = self.semnet(x)
        return res
     

def train_model(model_semnet, device, train_input_data, test_input_data, hyper_parameter, graph_parameter, user_parameter, save_net_folder, logs_file_name):  
    """
    Training the neural network (unchanged behavior).
    """

    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    batch_size, lr_enc, rnd_seed = hyper_parameter
    num_class, IR_num, split_type, out_norm = user_parameter

    IR_Str=format_IR(IR_num, split_type) # make a string 
    size_of_loss_check=1000
    
    optimizer_predictor = torch.optim.Adam(model_semnet.parameters(), lr=lr_enc)
    
    train_data=[]
    test_data=[]
    for ii in range(len(train_input_data)):
        train_data_tensor=torch.tensor(train_input_data[ii], dtype=torch.float).to(device)    
        train_data.append(train_data_tensor)
        test_data_tensor=torch.tensor(test_input_data[ii], dtype=torch.float).to(device)    
        test_data.append(test_data_tensor)        
   
    test_loss_total=[]
    train_loss_total=[]
    moving_avg=[]
    criterion = torch.nn.MSELoss()
    start_time=time.time()
    for iteration in range(5000000): # big upper bound with early stopping
        model_semnet.train()
        data_sets=train_data
        total_loss=0
        for idx_dataset in range(len(data_sets)):
            if len(data_sets[idx_dataset]) == 0:
                continue
            idx = torch.randint(0, len(data_sets[idx_dataset]), (batch_size,))
            data_train_samples = data_sets[idx_dataset][idx]
            calc_properties = model_semnet(data_train_samples).squeeze(1) 
            if num_class<=2:
                curr_pred_one_hot=torch.tensor([idx_dataset] * len(idx), dtype=torch.float).to(device)
            else:
                curr_pred=torch.tensor([idx_dataset] * len(idx), dtype=torch.long).to(device)
                curr_pred_one_hot = F.one_hot(curr_pred, num_classes=num_class).float().to(device)
            real_loss = criterion(calc_properties, curr_pred_one_hot)
            total_loss += torch.clamp(real_loss, min = 0., max = 50000.).double()

        optimizer_predictor.zero_grad()
        total_loss.backward()
        optimizer_predictor.step()
        
        # Evaluating the current quality.
        with torch.no_grad():
            model_semnet.eval()
            eval_datasets=flatten([train_data, test_data])
            all_real_loss=[]
      
            for idx_dataset in range(len(eval_datasets)):
                if len(eval_datasets[idx_dataset]) == 0:
                    all_real_loss.append(0.0)
                    continue
                take_n = min(size_of_loss_check, len(eval_datasets[idx_dataset]))
                calc_properties = model_semnet(eval_datasets[idx_dataset][0:take_n]).squeeze(1)       
                if num_class<=2:
                    curr_pred_one_hot=torch.tensor([idx_dataset%num_class] * take_n, dtype=torch.float).to(device)  
                else:
                    curr_pred=torch.tensor([idx_dataset%num_class] * take_n, dtype=torch.long).to(device)
                    curr_pred_one_hot = F.one_hot(curr_pred, num_classes=num_class).float().to(device)
                real_loss = criterion(calc_properties, curr_pred_one_hot)
                all_real_loss.append(float(real_loss.detach().cpu().numpy()))
             
            train_loss_total.append(sum(all_real_loss[:num_class]))
            test_loss_total.append(sum(all_real_loss[num_class:2*num_class]))

            if iteration%500==0:
                train_loss_number = sum(all_real_loss[:num_class])
                test_loss_number = sum(all_real_loss[num_class:2*num_class])

                print(f'    train_model: iteration: {iteration} - train loss: {train_loss_number}; test loss: {test_loss_number}; time: {time.time()-start_time}')
                with open(logs_file_name+"_logs.txt", "a") as myfile:
                    myfile.write(f'\n    train_model: iteration: {iteration} - train loss: {train_loss_number}; test loss: {test_loss_number}; time: {time.time()-start_time}')
                start_time=time.time()

            if len(test_loss_total) and test_loss_total[-1]==min(test_loss_total):
                model_semnet.eval()
                os.makedirs(save_net_folder, exist_ok=True)
                net_file=os.path.join(save_net_folder, f"net_full_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.pt")
                net_state_file=os.path.join(save_net_folder, f"net_state_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.pt")
                torch.save(model_semnet, net_file)
                torch.save(model_semnet.state_dict(), net_state_file)  
                model_semnet.train()
 
            if len(test_loss_total)>1000: # early stopping
                test_loss_moving_avg=sum(test_loss_total[-500:])
                moving_avg.append(test_loss_moving_avg)
                if len(moving_avg)>1000:
                    if moving_avg[-1]>moving_avg[-25] and moving_avg[-1]>moving_avg[-175] and moving_avg[-1]>moving_avg[-350] and moving_avg[-1]>moving_avg[-750] and moving_avg[-1]>moving_avg[-950]:
                        print('    Early stopping kicked in')
                        break
        
    return train_loss_total, test_loss_total, moving_avg


def plot_train_loss(train_loss_total, test_loss_total, moving_avg, graph_parameter, user_parameter, store_file):     
    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    num_class, IR_num, split_type, out_norm = user_parameter
   
    plt.figure()
    plt.plot(train_loss_total,label='train')
    plt.plot(test_loss_total,label='test')
    plt.title(f"test loss; ystart={year_start}, class={num_class}, IR={IR_num})")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs(os.path.dirname(store_file), exist_ok=True)
    plt.savefig(store_file, dpi=300)
    plt.show()
    plt.close()
        

def eval_model_in_batches(model, device, data_batch, data_feature, user_parameter):
    """
    Eval the trained neural network.
    Accepts numpy array or torch tensor. Returns numpy outputs (N,).
    """ 
    num_class, IR_num, split_type, out_norm = user_parameter
    
    # -------- PATCH: robust tensor creation --------
    if torch.is_tensor(data_feature):
        tensor_feature = data_feature.to(device)
    else:
        tensor_feature = torch.tensor(data_feature, dtype=torch.float, device=device)
        
    output_batches = []
    for start_i in range(0, len(tensor_feature), data_batch):
        end_i = start_i + data_batch
        batch_X = tensor_feature[start_i:end_i] 
        batch_output = model(batch_X).detach()
        output_batches.append(batch_output)
    outputs = torch.cat(output_batches).squeeze(1)  # (N,)
    outputs = outputs.cpu().numpy()
    
    if out_norm and num_class>2: # not used
        nn_outputs = F.softmax(torch.tensor(outputs), dim=0).numpy()
    else:
        nn_outputs=outputs
    return nn_outputs



def impact_classfication(full_train_data, data_feature_eval, solution_eval, pair_cf_parameter, hyper_parameter, graph_parameter, user_parameter, save_folders, logs_file_name):
    """
    Main training entry.
    - full_train_data: (N,3) [i,j, value]. value can be 0/1 (patched) or numeric citations (legacy).
    """ 

    node_cfeature_list, node_neighbor_list, num_neighbor_list, node_feature, node_cfeature=pair_cf_parameter 
    batch_size, lr_enc, rnd_seed = hyper_parameter
    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    num_class, IR_num, split_type, out_norm = user_parameter

    save_net_folder, save_loss_folder, save_figure_folder, save_result_folder= save_folders
    IR_Str=format_IR(IR_num, split_type)
    
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)   

    #####----------------------Prepare train and test data.-------------------------------####
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n1.1) {datetime.now()}: Prepare train and test data ...") 

    data_subset=prepare_split_datasets(full_train_data, user_parameter, logs_file_name)
    train_valid_test_size=[0.85, 0.15, 0.0]
    dataset_train, dataset_test=shuffle_split_datasets(data_subset, train_valid_test_size)
    
    pair_train, solution_train=get_pair_solution_datasets(dataset_train, hyper_parameter, user_parameter, logs_file_name)
    pair_test, solution_test=get_pair_solution_datasets(dataset_test, hyper_parameter, user_parameter, logs_file_name)
    
    ####--------train features------------####
    pair_feature_train, pair_cfeature_train=get_all_pair_features(node_cfeature_list, node_neighbor_list, num_neighbor_list, pair_train, logs_file_name)
    node_pair_feature_train=[node_feature, node_cfeature, pair_feature_train, pair_cfeature_train]
    data_feature_train, train_input_data = prepare_train_data(pair_train, solution_train, node_pair_feature_train, user_parameter, logs_file_name)
    
    ####--------test features------------####
    pair_feature_test, pair_cfeature_test=get_all_pair_features(node_cfeature_list, node_neighbor_list, num_neighbor_list, pair_test, logs_file_name)
    node_pair_feature_test=[node_feature, node_cfeature, pair_feature_test, pair_cfeature_test]
    data_feature_test, test_input_data = prepare_train_data(pair_test, solution_test, node_pair_feature_test, user_parameter, logs_file_name)

    #####----------------------Train Neural Network.-------------------------------####
    print(f"\n1.2) {datetime.now()}: Train Neural Network...") 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n1.2) {datetime.now()}: Train Neural Network...")     

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = len(data_feature_train[0]) # 141
    hidden_size = 600
    output_size = 1 if num_class<=2 else num_class
        
    model_semnet = ff_network(input_size, hidden_size, output_size).to(device)
    model_semnet.train() 
    train_loss, test_loss, moving_avg=train_model(model_semnet, device, train_input_data, test_input_data, hyper_parameter, graph_parameter, user_parameter, save_net_folder, logs_file_name)
    store_name=os.path.join(save_loss_folder, f"loss_plot_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.png")
    plot_train_loss(train_loss, test_loss, moving_avg, graph_parameter, user_parameter, store_name)
 
    #####------------AUC (train/test/eval).-----------------------####
    print(f'\n1.3) {datetime.now()}: Computes the AUC for training and test data...')
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n1.3) {datetime.now()}: Computes the AUC for training and test data...")

    model_semnet.eval()
    data_batch_size=batch_size
    
    roc_curve_name = [f"Train_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png",
                      f"Test_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png",
                      f"Eval_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png"]
    # train
    output_nn=eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_train, user_parameter)
    solution_arr=classify_solution(solution_train, user_parameter)
    train_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[0], save_figure_folder)
                
    # test
    output_nn=eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_test, user_parameter)
    solution_arr=classify_solution(solution_test, user_parameter)
    test_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[1], save_figure_folder)

    # eval future
    if len(data_feature_eval)>0: 
        print(f'\n1.4) {datetime.now()}: Evaluate the AUC for future data...')
        with open(logs_file_name+"_logs.txt", "a") as myfile:
            myfile.write(f"\n1.4) {datetime.now()}: Evaluate the AUC for future data...")

        output_nn=eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_eval, user_parameter)
        solution_arr=classify_solution(solution_eval, user_parameter)
        eval_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[2], save_figure_folder)
    else:
        eval_auc_score=[]
    
    store_folder=os.path.join(save_result_folder,f'AUC_Report_Year_{year_start}_Class_{num_class}_{IR_Str}.txt')
    with open(store_folder, 'a') as f:
        f.write(f"IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}\n")
    
    with open(os.path.join(save_result_folder, f"All_AUC_Report_Year_Train_{year_start}.txt"), 'a') as f:
        f.write(f"IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}\n")
        
    print(f"IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}")  
                
    return True
