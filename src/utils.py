from datasets import load_dataset
from collections import defaultdict
import os 
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm

DEBUG = False

def compute_accuracy(results_path_root, ds_conf, data_cache_dir, parser_fn_resp, parser_fn_correct, n_repeats=10, n_samples=None, return_subjects=True):
    """Return the accuracy per model reply for each seed"""

    accuracy = defaultdict(lambda:defaultdict(lambda:[]))
    # iterate over all the files under the root directory
    subjects = []
    for root, model_names, _ in os.walk(results_path_root):
        # get the name of each subdirectory under results_path_root
        
        for model_name in model_names:            
            # get the path of the subdirectory
            dir_path = os.path.join(root, model_name)
            # get the name of each file under the subdirectory
            for file in os.listdir(dir_path):
                if not file.endswith('.csv') or '900' in file:
                    continue
                seed = file.split('.')[0].split('seed')[-1]
                # get the path of the file
                file_path = os.path.join(dir_path, file)
                # read the file with the model replies
                model_replies = pd.read_csv(file_path)
                model_replies = model_replies.apply(lambda row: parser_fn_resp(row.iloc[0]), axis=1)
            
                # correct answers
                correct = load_dataset(ds_conf['dataset_path'], name=ds_conf['name'],split='test', cache_dir=data_cache_dir)
                correct.set_format(type='pandas')
                correct = correct[0:n_samples]
                correct = pd.DataFrame(np.repeat(correct.values, n_repeats, axis=0), columns=correct.columns)
                if return_subjects:
                    subjects = correct['subject']
                correct = correct.apply(lambda row: parser_fn_correct(row['answer']), axis=1)
                try:
                    accuracy[model_name][seed].extend(list(model_replies == correct))
                except Exception as e:
                    if DEBUG:
                        print(e)
                        print(model_replies)
                        print(correct)
                    pass
    # return the subject per question in the MMLU dataset
    if return_subjects:
        return accuracy, subjects
    return accuracy

def prepare_results_df(accuracy):
    """Return a DataFrame with the accuracy per model reply under coupled and independent generation"""
    results = pd.DataFrame(accuracy)
    results['Noise'] = results.index.map(lambda x: 'Coupled' if x == '72645763' else 'Independent')
    results.reset_index(inplace=True)
    results.drop('index', axis=1, inplace=True)
    results.set_index(['Noise'], inplace=True)
    results = results.groupby('Noise').agg('first')
    return results

 
def get_choice(s):
    """Parse the choice selected from the model"""
    choices = ['A', 'B', 'C', 'D']
    if not isinstance(s, str):
        if DEBUG:
            print('Non str found')
            print(s)
        s = str(s)
    # Split the string into words based on ' ' or new line characters
    words = s.split(' ')

    if not isinstance(words[0],str):
        if DEBUG:
            print('Non str found')
            print(words[0])
    
    # remove spurious characters from the choice
    choice = words[0].replace('.','').replace('"', '').replace(',','').replace(')','').replace(';', '').strip()
    
    if not any(choice==c for c in choices):
        choice = choice.split('\n')[0]

    if not  any(choice==c for c in choices):
        if DEBUG:
            print(choice)
        choice = 'E'
    
    return choice

def get_correct(choice_number):
    """Parse the correct choice from the dataset"""
    choices = ['A', 'B', 'C', 'D']
    return choices[int(choice_number)]

def metric_per_query(task, expl_results, metric='var'):
    """Return the variance of the scores difference (or the mean) per question for the given task (area)"""
    expl_results_task = expl_results[expl_results['Subject'] == task]
    expl_results_task = expl_results_task.drop('Subject', axis=1, inplace=False).reset_index(inplace=False)
    
    diffs = {}
    expl_results_task.set_index('Noise', inplace=True)
    for i, model_a in enumerate(expl_results_task.columns):
        for j, model_b in enumerate(expl_results_task.columns):
            if j <= i:
                continue
            a = expl_results_task[model_a]
            b = expl_results_task[model_b]
            diffs[(model_a, model_b)] = a - b 
            
    diffs_df = pd.DataFrame(diffs)
    query_id = [i  for i in range(len(expl_results_task)//20) for _ in range(10)]*2
    diffs_df['Qid'] = query_id
    var_per_query = diffs_df.groupby(['Qid', 'Noise']).agg(metric)
    pivoted = var_per_query.pivot_table(columns='Noise', index='Qid')
   
    return pivoted
    
    
def get_diffs_df(expl_results, task=None):
    diffs = {}
    if task:
        expl_results_task = expl_results[expl_results['Subject'] == task]
        expl_results_task = expl_results_task.drop('Subject', axis=1, inplace=False)
    else: 
        expl_results_task = expl_results
    expl_results_task.reset_index(inplace=False).set_index('Noise', inplace=True)
    for i, model_a in enumerate(expl_results_task.columns):
        for j, model_b in enumerate(expl_results_task.columns):
            if j <= i:
                continue
            a = expl_results_task[model_a]
            b = expl_results_task[model_b]
            diffs[(model_a, model_b)] = a - b 

    diffs_df = pd.DataFrame(diffs)
    return diffs_df


def calc_error(i, n_samples, results_dict, diffs_groupby_obj, reference_metric_fixed, reference_metric_ind):
    # get n samples for each model pair for each noise type
    sampled_diff = diffs_groupby_obj.sample(n_samples, random_state=i)
    sampled_metric_fixed = sampled_diff.iloc[:n_samples].mean()
    sampled_metric_ind = sampled_diff.iloc[n_samples:].mean()
    
    # concat dataframes using a new key

    results_dict[n_samples, 'Coupled', i] = abs(reference_metric_fixed - sampled_metric_fixed).to_dict()
    results_dict[n_samples, 'Independent', i] = abs(reference_metric_ind - sampled_metric_ind).to_dict()



def error_vs_samples(task, expl_results):
    n_repetitions=1000
    step=20
    diffs_df = get_diffs_df(expl_results, task=task)
    diffs_groupby_obj = diffs_df.groupby('Noise')


    manager = Manager()
    sampled_metric = manager.dict()
    reference_metric_fixed = diffs_df.iloc[:len(diffs_df)//2].mean()
    reference_metric_ind = diffs_df.iloc[len(diffs_df)//2:].mean()
    # create a pool of workers
    with Pool(20) as p:
        tqdm(p.starmap(calc_error, 
                    [(
                        i, 
                        n_samples, 
                        sampled_metric, 
                        diffs_groupby_obj, 
                        reference_metric_fixed, 
                        reference_metric_ind
                        ) for n_samples in range(step, len(diffs_df)//2, step) for i in range(n_repetitions)])
        )

    sampled_metric_df = pd.DataFrame(sampled_metric.copy())
    sampled_metric_df_T = sampled_metric_df.T

    # rename each index column
    sampled_metric_df_T.index.rename(['Samples', 'Noise', 'Iteration'], inplace=True)

    sampled_metric_df_T = sampled_metric_df_T.swaplevel('Samples', 'Noise', axis=0)
    sampled_metric_df_T = sampled_metric_df_T.swaplevel('Samples', 'Noise', axis=0)
    
    return sampled_metric_df_T

