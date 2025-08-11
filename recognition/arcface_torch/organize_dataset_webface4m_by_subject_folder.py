import os, sys
import argparse
import time
import glob



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/nobackup/unico/datasets/face_recognition/webface4m')
    parser.add_argument('--file-ext', type=str, default='.jpg')
    args = parser.parse_args()
    return args


def find_files_by_extension(folder_path, extension, ignore_file_with=''):
    matching_files = []
    num_files_found = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if ignore_file_with == '' or not ignore_file_with in file_path:
                    matching_files.append(file_path)
                    num_files_found += 1
                    print(num_files_found, end='\r')
    print()
    return sorted(matching_files)


def main(args):
    dataset_path = args.input_path.rstrip('/')
    output_path = os.path.join(os.path.dirname(dataset_path), f"{dataset_path.split('/')[-1]}_SUBJ_FOLDERS")

    print('dataset_path:', dataset_path)
    print('Searching samples...')
    ignore_file_with = '.cls'
    samples_paths = find_files_by_extension(dataset_path, args.file_ext, ignore_file_with)
    print(f'Found {len(samples_paths)} samples!')
    print('------')

    os.makedirs(output_path, exist_ok=True)
    print(f'Making symbolic links at: \'{output_path}\'')
    for idx_sample, sample_path in enumerate(samples_paths):
        subj_name = os.path.basename(sample_path).split('_')[0]
        print(f'Sample {idx_sample}/{len(samples_paths)} ({float(idx_sample)/float(len(samples_paths))*100.0:.1f}%) - Subj \'{subj_name}\'', end='\r')
        # print('subj_name:', subj_name)
        # sys.exit(0)
        subj_output_path = os.path.join(output_path, subj_name)
        sample_output_path = os.path.join(subj_output_path, os.path.basename(sample_path))
        os.makedirs(subj_output_path, exist_ok=True)
        
        os.symlink(sample_path, sample_output_path, target_is_directory=False)

    print('\nFinished!\n')



if __name__ == "__main__":
    args = parse_args()
    main(args)