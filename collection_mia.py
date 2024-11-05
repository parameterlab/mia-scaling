# %%
import logging

logging.basicConfig(level='ERROR')
import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_ind
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def combinations(list_items, num_combinations, num_groups):
    list_items_idx = list(range(len(list_items)))
    combinations = set()
    while len(combinations) < num_combinations:
        list_idx = random.sample(range(len(list_items_idx)), num_groups)
        combinations.add(tuple(list_idx))
    
    combinations_items = []
    for combination in combinations:
        combination_items = []
        for idx in combination:
            combination_items.append(list_items[idx])
        combinations_items.append(combination_items)
    return combinations_items
    # return list(combinations)

def process_combination_mia_features(list_combinations_mia_features):
    '''
    Convert a list of [{'pred': {'ppl': 5.1875,
                                'ppl/lowercase_ppl': -1.0285907103711933,
                                'ppl/zlib': 0.0001900983701566763,
                                'Min_5.0% Prob': 8.089154411764707,
                                'Min_10.0% Prob': 6.647058823529412,
                                'Min_20.0% Prob': 5.1873471882640585,
                                'Min_30.0% Prob': 4.264161746742671,
                                'Min_40.0% Prob': 3.5961376833740832,
                                'Min_50.0% Prob': 3.07814027370479,
                                'Min_60.0% Prob': 2.663736449002443},
                                'label': 0}, ... ]
    into a numpy array with all the features
    '''
    features = []
    for combination_mia_features in list_combinations_mia_features:
        dataset_features = []
        for mia_features in combination_mia_features:
            dataset_features.append(list(mia_features['pred'].values()))
        features.append(dataset_features)
        
    return np.array(features)

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    Train the model using the training set.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    criterion (torch.nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    num_epochs (int): Number of epochs to train the model. Default is 100.

    Returns:
    None
    """
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs).squeeze(1)  # Squeeze the output to match the shape of labels
            loss = criterion(outputs, labels.float())  # Convert labels to float
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    

def evaluate_model(model, eval_loader, threshold=0):
    model.eval()
    list_raw_scores = []
    list_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            predicted = (outputs.squeeze() > threshold).float()  # Apply threshold to get binary predictions
            try:
                list_raw_scores.extend(outputs.squeeze().detach().cpu().numpy().tolist())
                list_labels.extend(labels.detach().cpu().numpy().tolist())
                total += labels.size(0)
                correct += (predicted == labels.float()).sum().item()  # Convert labels to float for comparison
            except:
                print("Error")
                print(f"Outputs: {outputs}")

        accuracy = correct / total

    auc_score = roc_auc_score(list_labels, list_raw_scores)
    
    return accuracy, auc_score

# Example usage:
# accuracy, roc_auc = evaluate_model(model, eval_loader, threshold=0)
# print(f'Accuracy: {accuracy:.4f} with threshold {threshold}')
# print(f'ROC AUC: {roc_auc:.4f}')

def get_tpr_fpr(list_predictions, list_labels):
    tp = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 1 and list_labels[i] == 1])
    tn = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 0 and list_labels[i] == 0])
    fp = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 1 and list_labels[i] == 0])
    fn = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 0 and list_labels[i] == 1])
    fpr = fp / (fp + tn)*100
    tpr = tp / (tp + fn)*100
    return tpr, fpr

def get_raw_predictions(model, data_point):
    '''
    A data point is a set of docs/chunks, so a tensor of k x num_features
    '''
    model.eval()
    with torch.no_grad():
        outputs = model(torch.Tensor(data_point).to('cuda'))            
        return outputs.squeeze().detach().cpu().numpy()


def compute_auc(list_pvalues, list_labels, plot=False):
    list_tpr = []
    list_fpr = []
    for threshold in[1, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
        list_predictions = [1 if pval < threshold else 0 for pval in list_pvalues]
        tpr, fpr = get_tpr_fpr(list_predictions, list_labels)
        list_tpr.append(tpr/100)
        list_fpr.append(fpr/100)

    # sort ascending order of fpr
    # Sort list_fpr and list_tpr based on list_fpr
    sorted_indices = np.argsort(list_fpr)
    list_fpr = np.array(list_fpr)[sorted_indices]
    list_tpr = np.array(list_tpr)[sorted_indices]

    # compute auc using list_tpr and list_fpr
    auc_score = np.trapz(list_tpr, list_fpr)

    if plot:
        # plot ROC curve
        plt.plot(list_fpr, list_tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
    return auc_score

def run_dataset_inference_pvalues(args, num_docs_per_dataset, known_datasets):
    with open(os.path.join(args.mia_path, "mia_members.jsonl")) as f:
        mia_members = f.readlines()
        mia_members = [json.loads(x) for x in mia_members]

    with open(os.path.join(args.mia_path, "mia_nonmembers.jsonl")) as f:
        mia_non_members = f.readlines()
        mia_non_members = [json.loads(x) for x in mia_non_members]

    # members_text = load_from_disk(os.path.join(args.mia_path, "members"))
    # non_members_text = load_from_disk(os.path.join(args.mia_path, "nonmembers"))

    members_idx = list(range(len(mia_members)))
    non_members_idx = list(range(len(mia_non_members)))

    # shuffle the indices
    np.random.shuffle(members_idx)
    np.random.shuffle(non_members_idx)

    # shuffle mia_members and text in the same way
    mia_members = [mia_members[i] for i in members_idx]
    # members_text = members_text.select(members_idx)
    mia_non_members = [mia_non_members[i] for i in non_members_idx]
    # non_members_text = non_members_text.select(non_members_idx)
    # %%
    A_members = mia_members[:args.training_set_size_per_class]
    A_non_members = mia_non_members[:args.training_set_size_per_class]

    known_members = mia_members[args.training_set_size_per_class:args.training_set_size_per_class+known_datasets]
    known_non_members = mia_non_members[args.training_set_size_per_class:args.training_set_size_per_class+known_datasets]

    st_idx = args.training_set_size_per_class+known_datasets
    eval_members = mia_members[st_idx:]
    eval_non_members = mia_non_members[st_idx:]

    # replace al NaN by 0
    for mia in A_members + A_non_members + known_members + known_non_members + eval_members + eval_non_members:
        for key in mia['pred']:
            if np.isnan(mia['pred'][key]):
                mia['pred'][key] = 0

    # %%
    eval_members_datasets = combinations(eval_members, args.eval_set_size_per_class, num_docs_per_dataset)
    eval_non_members_datasets = combinations(eval_non_members, args.eval_set_size_per_class, num_docs_per_dataset)

    try:
        # known_members_datasets = combinations(known_members, len(known_members), num_docs_per_dataset)
        known_non_members_datasets = combinations(known_non_members, len(known_non_members), num_docs_per_dataset)
    except:
        print("Cannot use a full dataset as known dataset")
        known_non_members_datasets = combinations(known_non_members, len(known_non_members), len(known_non_members))

    # %%
    A_members = process_combination_mia_features([[x] for x in A_members])
    A_non_members = process_combination_mia_features([[x] for x in A_non_members])

    # known_members = process_combination_mia_features(known_members_datasets)
    known_non_members = process_combination_mia_features(known_non_members_datasets)

    eval_members_datasets = process_combination_mia_features(eval_members_datasets)
    eval_non_members_datasets = process_combination_mia_features(eval_non_members_datasets)

    # %%
    # remove the top 2.5% and bottom 2.5% of the training_members_datasets and training_non_members_datasets
    A_members = np.sort(A_members, axis=0)[int(0.025 * A_members.shape[0]):int(0.975 * A_members.shape[0])]
    A_non_members = np.sort(A_non_members, axis=0)[int(0.025 * A_non_members.shape[0]):int(0.975 * A_non_members.shape[0])]

    # known_members = np.sort(known_members, axis=0)[int(0.025 * known_members.shape[0]):int(0.975 * known_members.shape[0])]
    known_non_members = np.sort(known_non_members, axis=0)[int(0.025 * known_non_members.shape[0]):int(0.975 * known_non_members.shape[0])]

    # %%
    A_members.shape

    # %%
    print(f"Training size: {A_members.shape[0] + A_non_members.shape[0]}")
    print(f"Known size: {known_non_members.shape[0]}")
    print(f"Eval size: {eval_members_datasets.shape[0] + eval_non_members_datasets.shape[0]}")

    # %%
    training_set = np.concatenate([A_members, A_non_members], axis=0).squeeze(1) # (950, 10)
    training_labels = np.concatenate([np.ones(len(A_members)), np.zeros(len(A_non_members))]) # (2000,)

    eval_set = np.concatenate([eval_members_datasets, eval_non_members_datasets], axis=0)
    num_features = eval_set.shape[-1]
    eval_labels = np.concatenate([np.ones(len(eval_members_datasets)), np.zeros(len(eval_non_members_datasets))])

    # %%


    # %%
    # Convert to PyTorch tensors
    training_set = torch.tensor(training_set, dtype=torch.float32)
    training_labels = torch.tensor(training_labels, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(training_set, training_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # %%
    input_size = training_set.shape[1]
    hidden_size = input_size // 2
    output_size = 1  # membership score

    model = nn.Linear(input_size, 1).to('cuda')

    # %%
    # Step 4: Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    
    

    # %%
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    # %%
    # Chunk-level Evaluation
    B_members = process_combination_mia_features([[x] for x in eval_members])
    B_non_members = process_combination_mia_features([[x] for x in eval_non_members])
    chunk_eval_set = np.concatenate([B_members, B_non_members], axis=0).squeeze(1) # (950, 10)
    chunk_eval_labels = np.concatenate([np.ones(len(B_members)), np.zeros(len(B_non_members))]) # (2000,)
    # create Tensors
    chunk_eval_set = torch.tensor(chunk_eval_set, dtype=torch.float32)
    chunk_eval_labels = torch.tensor(chunk_eval_labels, dtype=torch.long)
    # Create DataLoader
    chunk_level_dataset = TensorDataset(chunk_eval_set, chunk_eval_labels)
    chunk_level_loader = DataLoader(chunk_level_dataset, batch_size=args.batch_size, shuffle=False)

    _, chunk_level_auc = evaluate_model(model, chunk_level_loader, threshold=0)



    # %%
    list_mia_distribution = []
    for data_point in eval_members_datasets:
        list_mia_distribution.append(get_raw_predictions(model, data_point))


    # %%
    # known_members_distribution = []
    # for data_point in known_members:
    #     known_members_distribution.append(get_raw_predictions(model, data_point))

    known_non_members_distribution = []
    for data_point in known_non_members:
        known_non_members_distribution.append(get_raw_predictions(model, data_point))

    # %%

    # %% [markdown]
    # # Single Test

    # %%
    

    # %%
    # members should be mapped to 1
    # non-members should be mapped to 0

    list_pvalues = []
    list_labels = []

    list_mia_distribution = []
    for data_point in eval_members_datasets:
        list_mia_distribution.append(get_raw_predictions(model, data_point))


    for dataset_distr in list_mia_distribution:
        statistic, pvalue = ttest_ind(dataset_distr,
                            np.array(known_non_members_distribution).reshape(-1),
                            equal_var=True,
                            alternative='greater')
        list_pvalues.append(statistic)
        list_labels.append(1)

    # count num pvalues < 0.05
    # num_significant = len([x for x in list_pvalues if x < 0.05])
    # print(f'Number of significant pvalues for members: {num_significant}; Percentage: {num_significant / len(list_pvalues) * 100:.2f}%')

    list_mia_distribution = []
    for data_point in eval_non_members_datasets:
        list_mia_distribution.append(get_raw_predictions(model, data_point))


    for dataset_distr in list_mia_distribution:
        statistic, pvalue = ttest_ind(dataset_distr,
                            np.array(known_non_members_distribution).reshape(-1),
                            equal_var=True,
                            alternative='greater')
        list_pvalues.append(statistic)
        list_labels.append(0)

    # num_significant = len([x for x in list_pvalues if x < 0.05])
    # print(f'Number of significant pvalues non-members: {num_significant}; Percentage: {num_significant / len(list_pvalues) * 100:.2f}%')
    

    auc_roc = roc_auc_score(list_labels, list_pvalues)
    # auc_roc = roc_auc_score([1 - x for x in list_labels], list_pvalues)
    fpr, tpr, thresholds = roc_curve(list_labels, list_pvalues)
    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]

    # list_predictions = [1 if pval < best_threshold else 0 for pval in list_pvalues]
    list_predictions = [1 if pval > best_threshold else 0 for pval in list_pvalues]
    tpr, fpr = get_tpr_fpr(list_predictions, list_labels)
    report = classification_report(list_labels, list_predictions, output_dict=True)
    best_f1 = report['weighted avg']['f1-score']


    print(f'Best threshold: {best_threshold}')
    print(f'Best F1: {100*best_f1:.4f} with threshold {best_threshold}; TPR: {tpr:.4f}, FPR: {fpr:.4f}; AUC: {auc_roc:.4f}; Chunk-level AUC: {chunk_level_auc:.4f}')
    # print(best_report)
    return best_f1*100, best_threshold, tpr, fpr, auc_roc, chunk_level_auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_size_per_class", type=int, default=1000)
    parser.add_argument("--known_datasets", type=int, default=None)
    parser.add_argument("--eval_set_size_per_class", type=int, default=1000)
    parser.add_argument("--num_docs_per_dataset", type=int, default=1000)

    parser.add_argument("--mia_path", type=str, default="out/dataset_mia/EleutherAI/pythia-2.8b/arxiv")
    
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    list_num_docs_per_dataset = [500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    for num_docs_per_dataset in list_num_docs_per_dataset:
        list_rows = []
        if args.known_datasets is None:
            for known_datasets in tqdm([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]):
                for seed in [670487, 116739, 26225, 777572, 288389]:
                    # set all random seeds
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    best_f1, best_pvalue, tpr, fpr, auc_roc, chunk_level_auc = run_dataset_inference_pvalues(args, num_docs_per_dataset, known_datasets) 
                    list_rows.append([num_docs_per_dataset, 
                                    known_datasets,
                                    args.training_set_size_per_class*2,
                                    args.eval_set_size_per_class*2,
                                    best_f1,
                                    best_pvalue,
                                    tpr,
                                    fpr,
                                    auc_roc,
                                    chunk_level_auc,
                                    seed,
                                    ])
                
        df = pd.DataFrame(list_rows, columns=['Dataset Size', 'Known Datasets', 'Training Size', 'Eval Size', 'F1', 'P-value', 'TPR', 'FPR', 'AUC', 'Chunk-level AUC', 'Seed'])
        df.to_csv(os.path.join(args.mia_path, f"dataset_inference_pvalues_{num_docs_per_dataset}_dataset_size.csv"), index=False)
                