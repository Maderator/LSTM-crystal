import os
import pickle


def load_logs_pickle(logs_path, pickle_filename: str = 'gridsearch_results.pkl'):
    results_path = os.path.join(logs_path, pickle_filename)
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def load_final_testing_experiments(
    experiment_path: str=os.path.join('experiments','09_test_dataset_eval'),
):
    # Only dynamic: layer1_dyn_results, layer10_results
    dyn_l1_path = os.path.join(experiment_path, 'logs', 'multi_layer_dynamic_1l_300')
    dyn_l2_10_path = os.path.join(experiment_path, 'logs', 'multi_layer_dynamic_10l')

    # All: layern_static_results
    static_l1_10_path = os.path.join(experiment_path, 'logs', 'multi_layer_static')
    

## BOOL FUNCTIONS ##

def has_bidi(exp):
    """Check if the experiment has bidirectional layers"""
    for key, value in exp.items():
        if key == "model-bidirectional" and value == True:
            return True
    return False

def has_dilation_base(exp):
    """Check if the experiment has a dilation base"""
    for key, value in exp.items():
        if key == "model-dilation_base" and value != None:
            return True
    return False

def has_res_block_size(exp):
    """Check if the experiment has a residual block size"""
    for key, value in exp.items():
        if key == "model-residual_block_size" and value != None:
            return True
    return False

def has_smyl_residual(exp):
    """Check if the experiment has a smyl (Kim et al.) residual"""
    for key, value in exp.items():
        if key == "model-smyl_residual" and value == True:
            return True
    return False 

def has_n_layers(exp, n):
    """Check if the experiment has n layers"""
    for key, value in exp.items():
        if key == "model-num_layers" and value == n:
            return True
    return False

### label functions ###

def get_experiment_label(exp):
    has_dynamic_lr = "training-additional_callbacks" in exp and exp["training-additional_callbacks"]
    #exp_name = f"l{exp["model-num_layers"]} u{exp["model-rnn_cell_params-units"]} {"lrDynamic" if has_dynamic_lr is not None else "lrStatic"}"
    exp_name = f"l{exp["model-num_layers"]} {"lrDynamic" if has_dynamic_lr is not None else "lrFixed"}"    
    return exp_name

### Results utils ###

def get_experiment_metrics_from_all_folds(experiment, loss_function_name="root_mean_squared_error", get_scaled=False, metric_summarization="metric_per_sample") -> list:
    metric_for_all_folds = []
    
    folds = experiment["metrics_per_fold"]
    for fold in folds:
        metrics = fold[loss_function_name]["scaled" if get_scaled else "unscaled"]
        metric_values = getattr(metrics, metric_summarization)
        metric_for_all_folds.extend(metric_values)
    return metric_for_all_folds

def get_experiments_samples(experiments, experiments_names, loss_function_name="root_mean_squared_error", get_scaled=False, metric_summarization="metric_per_sample") -> dict:
    experiments_samples = {}
    for exp, exp_name in zip(experiments, experiments_names):
        experiments_samples[exp_name] = get_experiment_metrics_from_all_folds(exp, loss_function_name=loss_function_name, get_scaled=get_scaled, metric_summarization=metric_summarization)
    return experiments_samples