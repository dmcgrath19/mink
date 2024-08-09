import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# helper functions
# def convert_huggingface_data_to_list_dic(dataset):
#     all_data = []
#     for i in range(len(dataset)):
#         ex = dataset[i]
#         all_data.append(ex)
#     return all_data
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset.iloc[i].to_dict()  # Convert row to dictionary
        all_data.append(ex)
    return all_data


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EleutherAI/pythia-2.8b')
# parser.add_argument(
#     '--dataset', type=str, default='WikiMIA_length32', 
#     choices=[
#         'WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128', 
#         'WikiMIA_length32_paraphrased',
#         'WikiMIA_length64_paraphrased',
#         'WikiMIA_length128_paraphrased', 
#     ]
# )
parser.add_argument('--dataset', type=str, default='WikiMIA_length32')
parser.add_argument('--perturbed_dataset', type=str, default='spanish_perturbed(150).csv')
parser.add_argument('--half', action='store_true')
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

# load model
def load_model(name):
    int8_kwargs = {}
    half_kwargs = {}
    # ref model is small and will be loaded in full precision
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    if 'mamba' in name:
        try:
            from transformers import MambaForCausalLM
        except ImportError:
            raise ImportError
        model = MambaForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

model, tokenizer = load_model(args.model)

# load dataset

dataset = pd.read_csv(args.dataset)
    # load_dataset('swj0419/WikiMIA', split=args.dataset)
# else: dataset = pd.read_csv('spanish_prompt(50).csv')
    
    # dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset)
data = convert_huggingface_data_to_list_dic(dataset)

perturbed_dataset = pd.read_csv(args.perturbed_dataset)


# load_dataset(
#     'zjysteven/WikiMIA_paraphrased_perturbed', 
#     split=args.dataset + '_perturbed'
# )
perturbed_data = convert_huggingface_data_to_list_dic(perturbed_dataset)
# print(perturbed_data[0])
num_neighbors = len(perturbed_data) // len(data)

# inference - get scores for each input
def inference(text, model, tokenizer):
    try:
        # Tokenize the input text
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ll = -loss.item()  # log-likelihood
        return ll
    
    except Exception as e:
        print(f"Skipping entry due to error: {e}")
        return None  # Indicate an error occurred

scores = defaultdict(list)
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    flagAdd = True
    text = d['input']
    ll = inference(text, model)

    ll_neighbors = []
    for j in range(num_neighbors):
        text = perturbed_data[i * num_neighbors + j]['input']
        inf_val = inference(text, model)
        if inf_val is not None:
            ll_neighbors.append(inf_val)
            flagAdd = False
            break

    # assuming the score is larger for training data
    # and smaller for non-training data
    # this is why sometimes there is a negative sign in front of the score
    if flagAdd:
        scores['neighbor'].append(ll - np.mean(ll_neighbors))
    else:
        flagAdd = True

# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df)

save_root = f"results/{args.model.split('/')[-1].split('.')[0]}"
if not os.path.exists(save_root):
    os.makedirs(save_root)

roc_data = {}
for scores, labels in scores.items():
    try:
        _, _, _, fpr, tpr = get_metrics(scores, labels)
        roc_data[method] = {
            'fpr': fpr,
            'tpr': tpr
        }
    except Exception as e:
        print(f"Error computing ROC data for method {scores}: {e}")

# Save ROC data to CSV for each method
for method, data in roc_data.items():
    df_roc = pd.DataFrame({
        'fpr': data['fpr'],
        'tpr': data['tpr']
    })
    roc_file = os.path.join(save_root, f"{method}_roc_data.csv")
    df_roc.to_csv(roc_file, index=False)

# Save combined ROC data to CSV
dataset_name = args.dataset.split('/')[-1].split('.')[0] if args.dataset and '/' in args.dataset else 'dataset'
model_id = args.model.split('/')[-1].split('.')[0]
combined_roc_file = os.path.join(save_root, f"{model_id}-{dataset_name}_roc_data.csv")

# Create a combined ROC DataFrame
combined_roc_data = pd.DataFrame()
for method, data in roc_data.items():
    df_roc = pd.DataFrame({
        'method': method,
        'fpr': data['fpr'],
        'tpr': data['tpr']
    })
    combined_roc_data = pd.concat([combined_roc_data, df_roc], ignore_index=True)

# Save combined ROC data to file
if not combined_roc_data.empty:
    combined_roc_data.to_csv(combined_roc_file, index=False)

# Append metrics to existing file or create new one
metrics_file = os.path.join(save_root, f"{model_id}.csv")
if os.path.isfile(metrics_file):
    df.to_csv(metrics_file, index=False, mode='a', header=False)
else:
    df.to_csv(metrics_file, index=False)