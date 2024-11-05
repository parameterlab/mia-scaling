# %%
import logging

logging.basicConfig(level='ERROR')
import argparse
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from scipy.stats import mannwhitneyu
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader, TensorDataset


# %%
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
    return np.array(combinations_items), combinations
    # return list(combinations)

# %%
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

# %%
import torch


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

# %%


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
            
            list_raw_scores.extend(outputs.squeeze().detach().cpu().numpy().tolist())
            list_labels.extend(labels.detach().cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()  # Convert labels to float for comparison

        accuracy = correct / total

    auc_score = roc_auc_score(list_labels, list_raw_scores)
    
    return accuracy, auc_score

# Example usage:
# accuracy, roc_auc = evaluate_model(model, eval_loader, threshold=0)
# print(f'Accuracy: {accuracy:.4f} with threshold {threshold}')
# print(f'ROC AUC: {roc_auc:.4f}')

def get_raw_predictions(model, docs_features):
    '''
    docs_features is a numpy array of shape (num_docs, num_features)
    '''
    model.eval()
    with torch.no_grad():
        outputs = model(docs_features.to('cuda')).cpu().numpy()
    return outputs

def get_mia_scorer(A_members, A_non_members, eval_members, eval_non_members, batch_size):
    # %%
    A_members = process_combination_mia_features([[x] for x in A_members])
    A_non_members = process_combination_mia_features([[x] for x in A_non_members])
    # remove the top 2.5% and bottom 2.5% of the training_members_datasets and training_non_members_datasets
    A_members = np.sort(A_members, axis=0)[int(0.025 * A_members.shape[0]):int(0.975 * A_members.shape[0])]
    A_non_members = np.sort(A_non_members, axis=0)[int(0.025 * A_non_members.shape[0]):int(0.975 * A_non_members.shape[0])]
    training_set = np.concatenate([A_members, A_non_members], axis=0).squeeze(1) # (950, 10)
    training_labels = np.concatenate([np.ones(len(A_members)), np.zeros(len(A_non_members))]) # (2000,)

    # Convert to PyTorch tensors
    training_set = torch.tensor(training_set, dtype=torch.float32)
    training_labels = torch.tensor(training_labels, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(training_set, training_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # %%
    print(f"Training size: {A_members.shape[0] + A_non_members.shape[0]}")

    # %%
    input_size = training_set.shape[1]
    hidden_size = input_size // 2
    output_size = 1  # membership score

    model = nn.Linear(input_size, 1).to('cuda')

    # %%
    # Step 4: Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    # %%
    # Chunk-level Evaluation
    B_members = process_combination_mia_features([[ch] for list_ch in eval_members for ch in list_ch])
    B_non_members = process_combination_mia_features([[ch] for list_ch in eval_non_members for ch in list_ch])
    # in some cases there could be some nan values for Min-k Prob, we will remove them (if it happens, it usually affect to the small probs)
    B_members = np.nan_to_num(B_members, nan=0.0)
    B_non_members = np.nan_to_num(B_non_members, nan=0.0)
    chunk_eval_set = np.concatenate([B_members, B_non_members], axis=0).squeeze(1) # (950, 10)
    chunk_eval_labels = np.concatenate([np.ones(len(B_members)), np.zeros(len(B_non_members))]) # (2000,)
    # create Tensors
    chunk_eval_set = torch.tensor(chunk_eval_set, dtype=torch.float32)
    chunk_eval_labels = torch.tensor(chunk_eval_labels, dtype=torch.long)
    # Create DataLoader
    chunk_level_dataset = TensorDataset(chunk_eval_set, chunk_eval_labels)
    chunk_level_loader = DataLoader(chunk_level_dataset, batch_size=batch_size, shuffle=False)

    accuracy, roc_auc_chunk = evaluate_model(model, chunk_level_loader, threshold=0)
    return model, roc_auc_chunk

# %%
def dict_to_tensor(mia_features):
    # Extract the 'pred' values and convert them to a list
    pred_values = list(mia_features['pred'].values())
    # clean nan values (replace with 0)
    pred_values = [0 if np.isnan(x) else x for x in pred_values]
    # Convert the list to a tensor
    tensor = torch.tensor(pred_values)
    
    return tensor

def convert_docs2tensors(docs):
    '''
    Convert a list of documents (dict of mia features) to a list of tensors. A tensor for each document (# paragraphs x #mia_features)
    '''
    tensors = []
    for doc in docs:
        paragraph_tensor = []
        for paragraph in doc:
            paragraph_tensor.append(dict_to_tensor(paragraph))
        tensors.append(torch.stack(paragraph_tensor))
    return tensors

# %%
def get_eval_scores(model, members_path, non_members_path):
    with open(members_path) as f:
        eval_members = f.readlines()
        eval_members = [json.loads(x) for x in eval_members]

    with open(non_members_path) as f:
        eval_non_members = f.readlines()
        eval_non_members = [json.loads(x) for x in eval_non_members]

    eval_members_tensor = convert_docs2tensors(eval_members)
    eval_non_members_tensor = convert_docs2tensors(eval_non_members)

    eval_members_docs_scores = []
    for tensor in eval_members_tensor:
        eval_members_docs_scores.append(get_raw_predictions(model, tensor))

    eval_non_members_docs_scores = []
    for tensor in eval_non_members_tensor:
        eval_non_members_docs_scores.append(get_raw_predictions(model, tensor))
    
    return eval_members_docs_scores, eval_non_members_docs_scores


# %%
def evaluate(eval_members_docs_scores, eval_non_members_docs_scores, known_non_members_docs_scores, num_paragraphs):
    list_scores = []
    list_labels = []
    for eval_member_score in eval_members_docs_scores:
        if len(eval_member_score) > 1:
            statistic, pvalue = mannwhitneyu(eval_member_score[:num_paragraphs].squeeze(),
                                known_non_members_docs_scores,
                                alternative='greater')
            # statistic, pvalue = ttest_ind(eval_member_score[:num_paragraphs].squeeze(),
            #                     known_non_members_docs_scores,
            #                     alternative='greater',
            #                     equal_var=False)
            if not np.isnan(statistic):
                list_scores.append(statistic)
                list_labels.append(1)


    for eval_non_member_score in eval_non_members_docs_scores:
        if len(eval_non_member_score) > 1:
            statistic, pvalue = mannwhitneyu(eval_non_member_score[:num_paragraphs].squeeze(),
                                known_non_members_docs_scores,
                                alternative='greater')
            # statistic, pvalue = ttest_ind(eval_non_member_score[:num_paragraphs].squeeze(),
            #                     known_non_members_docs_scores,
            #                     alternative='greater',
            #                     equal_var=False)
            if not np.isnan(statistic):
                list_scores.append(statistic)
                list_labels.append(0)

    auc_roc = roc_auc_score(list_labels, list_scores)
    fpr, tpr, thresholds = roc_curve(list_labels, list_scores)
    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]

    list_predictions = [1 if pval > best_threshold else 0 for pval in list_scores]
    tpr, fpr = get_tpr_fpr(list_predictions, list_labels)
    report = classification_report(list_labels, list_predictions, output_dict=True)
    best_f1 = report['weighted avg']['f1-score']

    return best_f1, tpr, fpr, auc_roc
# %%
def get_tpr_fpr(list_predictions, list_labels):
    tp = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 1 and list_labels[i] == 1])
    tn = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 0 and list_labels[i] == 0])
    fp = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 1 and list_labels[i] == 0])
    fn = sum([1 for i in range(len(list_predictions)) if list_predictions[i] == 0 and list_labels[i] == 1])
    fpr = fp / (fp + tn)*100
    tpr = tp / (tp + fn)*100
    return tpr, fpr

def clean_nan(mia_partition):
    for list_sents in mia_partition:
        for sent in list_sents:
            for (mia, value) in sent['pred'].items():
                if np.isnan(value):
                    sent['pred'][mia] = 100000

def run_mia(mia_path, training_set_size_per_class, known_datasets, batch_size):
    with open(os.path.join(mia_path, "mia_members.jsonl")) as f:
        mia_members = f.readlines()
        mia_members = [json.loads(x) for x in mia_members]

    with open(os.path.join(mia_path, "mia_nonmembers.jsonl")) as f:
        mia_non_members = f.readlines()
        mia_non_members = [json.loads(x) for x in mia_non_members]

    clean_nan(mia_members)
    clean_nan(mia_non_members)
    # %%
    members_text = load_from_disk(os.path.join(mia_path, "members"))
    non_members_text = load_from_disk(os.path.join(mia_path, "nonmembers"))

    # %%
    members_idx = list(range(len(mia_members)))
    non_members_idx = list(range(len(mia_non_members)))

    # shuffle the indices
    np.random.shuffle(members_idx)
    np.random.shuffle(non_members_idx)

    # shuffle mia_members and text in the same way
    mia_members = [mia_members[i] for i in members_idx]
    members_text = members_text.select(members_idx)
    mia_non_members = [mia_non_members[i] for i in non_members_idx]
    non_members_text = non_members_text.select(non_members_idx)
    

    A_non_members = [x for list_paragraphs in mia_non_members[:950] for x in list_paragraphs]
    known_non_members = [x for list_paragraphs in mia_non_members[950:1000] for x in list_paragraphs]
    eval_non_members = mia_non_members[1000:]

    A_members = [x for list_paragraphs in mia_members[:950] for x in list_paragraphs ]
    known_members = [x for x in mia_members[950:1000]]
    eval_members = mia_members[1000:]

    print(f"Num. training paragraphs: {len(A_members) + len(A_non_members)}")
    print(f"Num. known docs: {len(known_members)}")
    print(f"Num. member eval docs: {len(eval_members)}")
    print(f"Num. non-member eval docs: {len(eval_non_members)}")

    eval_set_size = len(eval_members) + len(eval_non_members)


    # %% [markdown]
    # # Step 1: Train Chunk-level Model
    model, roc_auc_chunk = get_mia_scorer(A_members, A_non_members, eval_members, eval_non_members, batch_size)
    print(f'Chunk-level ROC AUC: {roc_auc_chunk:.4f}')
    
    # %% [markdown]
    # # Step 2: Prepare Document-level Evaluation

    # %%
    eval_members_tensor = convert_docs2tensors(eval_members)
    eval_non_members_tensor = convert_docs2tensors(eval_non_members)

    eval_members_docs_scores = []
    for tensor in eval_members_tensor:
        eval_members_docs_scores.append(get_raw_predictions(model, tensor))

    eval_non_members_docs_scores = []
    for tensor in eval_non_members_tensor:
        eval_non_members_docs_scores.append(get_raw_predictions(model, tensor))

    

    # %% [markdown]
    # # Step 3: Process the Known Partition

    # %%
    known_non_members_ = process_combination_mia_features([[x] for x in known_non_members])
    known_non_members_ = np.nan_to_num(known_non_members_, nan=0.0)
    known_non_members_tensor = torch.tensor(known_non_members_, dtype=torch.float32)
    known_non_members_docs_scores = get_raw_predictions(model, known_non_members_tensor)
    known_non_members_docs_scores = np.array([x for l in known_non_members_docs_scores for x in l]).reshape(-1)

    ## Run Statistic Tests
    # list_num_paragraphs = (set([len(x) for x in eval_members_docs_scores + eval_non_members_docs_scores]))
    # list_num_paragraphs = [x for x in list_num_paragraphs if x > 1]
    list_num_paragraphs = [len(x) for x in eval_members_docs_scores + eval_non_members_docs_scores]
    print(f"Running mia for up to {int(np.mean(list_num_paragraphs) + np.std(list_num_paragraphs))} paragraphs")
    list_auc_roc = []
    list_paragraphs = list(range(2, int(np.mean(list_num_paragraphs) + np.std(list_num_paragraphs))))
    for num_paragraphs in list_paragraphs:
        best_f1, tpr, fpr, auc_roc = evaluate(eval_members_docs_scores, eval_non_members_docs_scores, known_non_members_docs_scores, num_paragraphs)
        list_auc_roc.append(auc_roc)

    return list_paragraphs, list_auc_roc, roc_auc_chunk, eval_set_size, np.mean(list_num_paragraphs)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--training_set_size_per_class", type=int, default=2000)
    parser.add_argument("--known_datasets", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_path", type=str, default="results/doc_mia/EleutherAI")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # %%
    # generate 5 random seeds
    seeds = [random.randint(0, 100000) for _ in range(5)]
    for seed in seeds:
        set_seed(seed)
        # %%
        list_chunk_sizes = [512, 1024, 2048]
        list_llm_names = [os.path.basename(x) for x in glob.glob(os.path.join(args.base_path, "*"))]
        for llm_name in list_llm_names:
            model_size = llm_name.split("pythia-")[1].split("/")[0]
            for dataset_path in glob.glob(f"{args.base_path}/{llm_name}/haritzpuerto/*"):
                dataset_name = os.path.basename(dataset_path).split("_")[-1]
                for chunk_size in list_chunk_sizes:
                    full_mia_path = os.path.join(dataset_path, f"sample_size_2000/{chunk_size}")
                    print(f"Running {full_mia_path}. Seed {seed}")
                    try:
                        list_num_paragraphs, list_auc_roc, roc_auc_chunk, eval_set_size, avg_paragraphs = run_mia(full_mia_path, args.training_set_size_per_class, args.known_datasets, args.batch_size)
                        df = pd.DataFrame({"num_paragraphs": list_num_paragraphs,
                                        "auc_roc": list_auc_roc,
                                        "roc_auc_chunk": [roc_auc_chunk]*len(list_num_paragraphs),
                                        "eval_set_size": [eval_set_size]*len(list_num_paragraphs),
                                        "seed": [seed]*len(list_num_paragraphs),
                                        "avg_paragraphs": [avg_paragraphs]*len(list_num_paragraphs),
                                        })
                        output_path = os.path.join(full_mia_path, "mia_results_950-50-1000")
                        os.makedirs(output_path, exist_ok=True)
                        df.to_csv(os.path.join(output_path, f"mia_results_{seed}.csv"), index=False)
                        plt.plot(list_num_paragraphs, list_auc_roc, label=f'{model_size} - {chunk_size}')
                        plt.xlabel('# Paragraphs')
                        plt.ylabel('AUC')
                        plt.title(f'AUC vs. # Paragraphs ({dataset_name}) - Chunk-level AUC: {roc_auc_chunk:.4f}')
                        plt.legend(loc='lower right')
                        # save the plot
                        plt.savefig(os.path.join(output_path, f"mia_results_{seed}.png"))
                        plt.close()
                        print(f"Done running {output_path}")
                    except Exception as e:
                        print(f"Error running {output_path}")
                        print(e)
                        with open(f"{args.base_path}/{llm_name}/haritzpuerto/errors.txt", "w") as f:
                            f.write(str(e))
                        continue
