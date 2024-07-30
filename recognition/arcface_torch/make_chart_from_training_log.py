import sys, os
import argparse
import re
from datetime import datetime


def parse_insightface_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    hyperparameters = {}
    training_values = []
    validation_values = []

    # Regex patterns to match lines
    hyperparam_pattern = re.compile(r'Training: [0-9]*-[0-9]*-[0-9]* [0-9]*:[0-9]*:[0-9]*,[0-9]*-:\s')
    training_pattern = re.compile(r'Training: [0-9]*-[0-9]*-[0-9]*\s[0-9]*:[0-9]*:[0-9]*,[0-9]*-.*Loss.*LearningRate')
    validation_pattern = re.compile(r'Training: [0-9]*-[0-9]*-[0-9]*\s[0-9]*:[0-9]*:[0-9]*,[0-9]*-.*]Accuracy-Flip')

    for line in lines:
        line = line.strip()

        # Match hyperparameters
        if hyperparam_pattern.match(line):
            sub_line = re.sub('Training: [0-9]*-[0-9]*-[0-9]* [0-9]*:[0-9]*:[0-9]*,[0-9]*-:\s', '', line)
            sub_line = sub_line.replace(', ', ',')
            sub_line_split = sub_line.split(' ')
            sub_line_split = [value for value in sub_line_split if value != '']
            if len(sub_line_split) > 2:
                sub_line_split = [sub_line_split[0]] + [value for value in sub_line_split[1:]]
            if len(sub_line_split) > 1:
                hyperparameters[sub_line_split[0]] = sub_line_split[1]

        # Match training values
        elif training_pattern.match(line):
            sub_line = line.replace('Training: ', '')
            sub_line = re.sub(',[0-9]*\-', '   ', sub_line)
            sub_line = sub_line.replace(' samples/sec', '')
            sub_line = sub_line.replace(' hours', '')
            sub_line_split = sub_line.split('   ')

            training_values.append({
                "datetime":        sub_line_split[0],
                "speed":           float(sub_line_split[1].split(' ')[-1]),
                "loss":            float(sub_line_split[2].split(' ')[-1]),
                "learning_rate":   float(sub_line_split[3].split(' ')[-1]),
                "epoch":           int(sub_line_split[4].split(' ')[-1]),
                "global_step":     int(int(sub_line_split[5].split(' ')[-1])),
                "fp16_grad_scale": int(sub_line_split[6].split(' ')[-1]),
                "required_time":   int(sub_line_split[7].split(' ')[-1]),
            })

        # Match validation values
        elif validation_pattern.match(line):
            sub_line = re.sub('Training: [0-9]*-[0-9]*-[0-9]*\s[0-9]*:[0-9]*:[0-9]*,[0-9]*-', '', line)
            sub_line = sub_line.replace('[', '').replace(']', ' ').replace(':', '').replace('+-', ' ')
            sub_line_split = sub_line.split(' ')

            validation_values.append({
                "dataset":       sub_line_split[0].split('/')[-1].replace('.bin', ''),
                "epoch":         int(training_values[-1]['epoch']),
                "global_step":   int(sub_line_split[1]),
                "accuracy_flip": float(sub_line_split[3]),
                "std":           float(sub_line_split[4])
            })

    return hyperparameters, training_values, validation_values


def group_logs_by_epoch(logs_dict=[{}]):
    keys_logs = list(logs_dict[0].keys())
    epoch = -1
    epochs_logs = {}

    for idx_log, log in enumerate(logs_dict):
        if log['epoch'] > epoch:   # start epoch line
            epoch = log['epoch']
            epochs_logs[epoch] = {key:[] for key in keys_logs}
        for key in keys_logs:
            epochs_logs[epoch][key].append(log[key])

    for idx_epoch, epoch in enumerate(epochs_logs.keys()):
        num_steps_epoch = len(epochs_logs[epoch][keys_logs[0]])
        for idx_key, key in enumerate(keys_logs[1:]):
            assert len(epochs_logs[epoch][key]) == num_steps_epoch, f'Error, len(epochs_logs[epoch][key]) ({len(epochs_logs[epoch][key])}) != num_steps_epoch ({num_steps_epoch}). Must be equal!'

    return epochs_logs


def organize_val_logs_by_dataset(logs_dict=[{}]):
    epochs = list(logs_dict.keys())
    keys_logs = list(logs_dict[0].keys())
    datasets = []
    for dataset in logs_dict[0]['dataset']:
        if not dataset in datasets:
            datasets.append(dataset)
    logs_by_datasets = {}

    # print('datasets:', datasets)
    # sys.exit(0)

    for idx_dataset, dataset in enumerate(datasets):
        logs_one_dataset = {'epochs': [], 'accuracy_flip': []}
        for epoch in epochs:
            # print('idx_dataset:', idx_dataset, '    dataset:', dataset, '    epoch:', epoch, '    accuracy_flip:', logs_dict[epoch]['accuracy_flip'][-len(datasets):][idx_dataset])
            logs_one_dataset['epochs'].append(epoch)
            logs_one_dataset['accuracy_flip'].append(logs_dict[epoch]['accuracy_flip'][-len(datasets):][idx_dataset])
        # sys.exit(0)
        logs_by_datasets[dataset] = logs_one_dataset
        # print(f"logs_by_datasets['{dataset}']: {logs_by_datasets[dataset]}")
    return logs_by_datasets
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, default='recognition/arcface_torch/work_dirs/casia_dcface_frcsyn_r100/2023-10-28_16-20-37_GPU0/training.log')
    args = parser.parse_args()

    print(f'Loading logs from \'{args.log_path}\'')
    hyperparams, training_logs, validation_logs = parse_insightface_log_file(args.log_path)
    print('Done!')
    # print("Hyperparameters:")
    # print(hyperparams)
    # print("\nTraining Data:")
    # for data in training_data:
    #     print(data)
    # print("\nValidation Data:")
    # for data in validation_data:
    #     print(data)

    print(f'\nGrouping logs by epoch...')
    training_logs_epochs = group_logs_by_epoch(training_logs)
    # print('training_logs_epochs:', training_logs_epochs)
    validation_logs_epochs = group_logs_by_epoch(validation_logs)
    print('Done!')

    # print('\nvalidation_logs_epochs:', validation_logs_epochs)
    # print('\nvalidation_logs_epochs:', validation_logs_epochs[0])


    validation_logs_by_datasets = organize_val_logs_by_dataset(validation_logs_epochs)
    # datasets = list(validation_logs_by_datasets.keys())
    # for dataset in datasets:
    #     print('dataset:', dataset, '    ', validation_logs_by_datasets[dataset])