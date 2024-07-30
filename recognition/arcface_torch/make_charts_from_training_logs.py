import sys, os
import argparse
import re
from datetime import datetime
import matplotlib.pyplot as plt


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


def organize_all_val_logs_by_dataset(logs_dict=[{}]):
    train_datasets = list(logs_dict.keys())
    val_datasets = list(logs_dict[train_datasets[0]].keys())
    # print('train_datasets:', train_datasets)
    # print('val_datasets:', val_datasets)

    val_logs_by_dataset = {}
    for idx_val_dataset, val_dataset in enumerate(val_datasets):
        logs_one_val_dataset = {}
        for idx_train_dataset, train_dataset in enumerate(train_datasets):
            logs_one_val_dataset[train_dataset] = logs_dict[train_dataset][val_dataset]
            # print(f"{val_dataset} - {train_dataset} - {logs_dict[train_dataset][val_dataset]}")
            # print('---')
        # print(f"{val_dataset} - {logs_one_val_dataset}")
        val_logs_by_dataset[val_dataset] = logs_one_val_dataset
    return val_logs_by_dataset


def plot_accuracy_curves(data, title, file_path):
    plt.figure(figsize=(6, 5))
    
    # Iterate over each dataset in the dictionary
    for dataset, values in data.items():
        epochs = values['epochs']
        accuracies = values['accuracy_flip']
        plt.plot(epochs, accuracies, label=dataset)
    
    interval = 5
    epochs_ticks = []
    for epoch in epochs:
        if epoch == 0 or (epoch+1)%interval == 0:
            epochs_ticks.append(epoch)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='best')
    plt.xticks(epochs_ticks)
    plt.xlim([0, 19])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    
    # Save the plot in both SVG and PNG formats
    plt.savefig(file_path)
    plt.savefig(os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace('.png', '.svg')))
    # plt.show()



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log-path", type=str, default='recognition/arcface_torch/work_dirs/casia_dcface_frcsyn_r100/2023-10-28_16-20-37_GPU0/training.log')
    # args = parser.parse_args()

    logs_paths = [
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/CASIA_r100_aug_one_gpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/DCFace_r100_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/DigiFace_r100_aug_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/GANDiffFace_r100_aug_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/IDiff-Face_r100_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/idnet_r100_aug_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/SFace_r100_aug_onegpu/training.log',
        '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_pedro/SynFace_r100_aug_onegpu/training.log',  
    ]

    train_datasets_logs = {}
    for idx_log, log_path in enumerate(logs_paths):
        train_dataset = log_path.split('/')[-2].split('_')[0]

        print(f'train_dataset: {train_dataset} - Loading logs from \'{log_path}\'')
        hyperparams, training_logs, validation_logs = parse_insightface_log_file(log_path)

        # training_logs_epochs = group_logs_by_epoch(training_logs)
        validation_logs_epochs = group_logs_by_epoch(validation_logs)
        
        validation_logs_by_datasets = organize_val_logs_by_dataset(validation_logs_epochs)
        # datasets = list(validation_logs_by_datasets.keys())
        # for dataset in datasets:
        #     print('dataset:', dataset, '    ', validation_logs_by_datasets[dataset])

        train_datasets_logs[train_dataset] = validation_logs_by_datasets
        # print(f"train_datasets_logs[{train_dataset}]:", train_datasets_logs[train_dataset])

    all_validation_logs_by_datasets = organize_all_val_logs_by_dataset(train_datasets_logs)
    val_datasets = list(all_validation_logs_by_datasets.keys())
    # print('val_datasets:', val_datasets)
    # print('val_datasets[0]:', val_datasets[0], ' - ', all_validation_logs_by_datasets[val_datasets[0]])

    for val_dataset in val_datasets:
        title = val_dataset
        chart_path = os.path.join('/'.join(logs_paths[0].split('/')[:-2]), f"{val_dataset}.png")
        print('Saving chart:', chart_path)
        plot_accuracy_curves(all_validation_logs_by_datasets[val_dataset], title, chart_path)