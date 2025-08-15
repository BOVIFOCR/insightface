import os
import sys
import argparse
import subprocess
import pandas as pd
import re
from pathlib import Path
from io import StringIO
import json
import csv
import socket



hostname = socket.gethostname()

if 'duo' in hostname:
    benchmarks = [('hda_doppelganger',       '--network %s --model %s --target hda_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --facial-attributes /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB --data-dir2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs --facial-attributes2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs_FACE_ATTRIB'),
                  ('doppelver_doppelganger', '--network %s --model %s --target doppelver_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/DoppelgangerProtocol.csv --ignore-missing-imgs --facial-attributes /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                  ('doppelver_vise',         '--network %s --model %s --target doppelver_vise --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/ViSEProtocol.csv --ignore-missing-imgs --facial-attributes /datasets1/bjgbiesseck/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                  ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC/exp1_gallery.txt --facial-attributes /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
                  ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC/exp3_gallery.txt --facial-attributes /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
                  ('nd_twins',               '--network %s --model %s --target nd_twins --data-dir /datasets1/bjgbiesseck/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs --protocol /datasets1/bjgbiesseck/ND-Twins-2009-2010/TwinsChallenge_1.0.0/TwinsChallenge/data/TwinsPairTable.csv --facial-attributes /datasets1/bjgbiesseck/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                 ]

    # benchmarks = [('hda_doppelganger',       '--network %s --model %s --target hda_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --facial-attributes /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB --data-dir2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs --facial-attributes2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs_FACE_ATTRIB'),
    #               ('3d_tec_exp1',            '--network %s --model %s --target 3d_tec --data-dir /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC/exp1_gallery.txt --facial-attributes /datasets1/bjgbiesseck/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB')
    #               ]

    # benchmarks = [('hda_doppelganger',       '--network %s --model %s --target hda_doppelganger --data-dir /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --facial-attributes /datasets1/bjgbiesseck/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB --data-dir2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs --facial-attributes2 /datasets1/bjgbiesseck/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs_FACE_ATTRIB')
    #               ]

elif 'diolkos' in hostname:
    # benchmarks = [('hda_doppelganger',       '--network %s --model %s --target hda_doppelganger --data-dir /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs --facial-attributes /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/HDA-Doppelgaenger_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB --data-dir2 /experiments/bjgbiesseck/datasets/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs --facial-attributes2 /experiments/bjgbiesseck/datasets/MICA/FRGC/images_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs_FACE_ATTRIB'),
    #               ('doppelver_doppelganger', '--network %s --model %s --target doppelver_doppelganger --data-dir /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/DoppelgangerProtocol.csv --ignore-missing-imgs --facial-attributes /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
    #               ('doppelver_vise',         '--network %s --model %s --target doppelver_vise --data-dir /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/ViSEProtocol.csv --ignore-missing-imgs --facial-attributes /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
    #               ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC/exp1_gallery.txt --facial-attributes /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
    #               ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC/exp3_gallery.txt --facial-attributes /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
    #               ('nd_twins',               '--network %s --model %s --target nd_twins --data-dir /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs --protocol /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/TwinsChallenge_1.0.0/TwinsChallenge/data/TwinsPairTable.csv --facial-attributes /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs_FACE_ATTRIB'),
    #              ]

    benchmarks = [('doppelver_doppelganger', '--network %s --model %s --target doppelver_doppelganger --data-dir /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/DoppelgangerProtocol.csv --ignore-missing-imgs --facial-attributes /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                  ('doppelver_vise',         '--network %s --model %s --target doppelver_vise --data-dir /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_EMBEDDINGS_OUTLIERS_INLIERS/thresh=0.4/inliers --protocol /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/ViSEProtocol.csv --ignore-missing-imgs --facial-attributes /experiments/bjgbiesseck/datasets/doppelgangers_lookalikes/DoppelVer/Images/CCA_Images_DETECTED_FACES_RETINAFACE_scales=[1.0,0.5,0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                  ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC/exp1_gallery.txt --facial-attributes /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
                  ('3d_tec',                 '--network %s --model %s --target 3d_tec --data-dir /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages --protocol /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC/exp3_gallery.txt --facial-attributes /experiments/bjgbiesseck/datasets/3D-Twins-Expression-Challenge_3D-TEC_DETECTED_FACES_RETINAFACE_scales=[0.5]_nms=0.4/imgs/textureimages_FACE_ATTRIB'),
                  ('nd_twins',               '--network %s --model %s --target nd_twins --data-dir /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs --protocol /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/TwinsChallenge_1.0.0/TwinsChallenge/data/TwinsPairTable.csv --facial-attributes /experiments/bjgbiesseck/datasets/ND-Twins-2009-2010/images_DETECTED_FACES_RETINAFACE_scales=[0.25]_nms=0.4/imgs_FACE_ATTRIB'),
                 ]
else:
    raise Exception(f'No benchmarks defined for this host \'{hostname}\'')





models = [('R100/WebFace4M',     'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs2/webface4m_r100_aug_onegpu/model.pt'),
          ('R100/MS1MV3',        'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r100_fp16/backbone.pth'),
          ('R100/Glint360K',     'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/glint360k_cosface_r100_fp16_0.1/backbone.pth'),
          ('R100/CASIA-Webface', 'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt'),
          ('R100/DCFace',        'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs/dcface_frcsyn_r100/2023-10-20_16-50-08_GPU0/model.pt'),
          ('R100/IDiffFace',     'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_synthetic/IDiff-Face_r100_onegpu/model.pt'),
          ('R100/IDNet',         'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_synthetic/idnet_r100_aug_onegpu/model.pt'),
          ('R100/SFace',         'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_synthetic/SFace_r100_aug_onegpu/model.pt'),
          ('R100/GANDiffFace',   'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs/gandiffface_frcsyn_r100/2023-10-19_19-48-47_GPU0/model.pt'),
          ('R100/DigiFace',      'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_synthetic/DigiFace_r100_aug_onegpu/model.pt'),
          ('R100/SynFace',       'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs_synthetic/SynFace_r100_aug_onegpu/model.pt'),
          ('R100/DisCo',         'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs2/disco_r100_aug_one_gpu/model.pt'),
          ('R100/Arc2Face',      'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs2/arc2face_r100_onegpu/model.pt'),
          ('R100/VIGFace',       'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs2/vigface_r100_aug_one_gpu/model.pt')
          ]

# models = [('R100/CASIA-Webface', 'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt'),
#           ('R100/MS1MV3',        'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/ms1mv3_arcface_r100_fp16/backbone.pth'),
#           ('R100/Glint360K',     'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/trained_models/glint360k_cosface_r100_fp16_0.1/backbone.pth')
#           ]

# models = [('R100/CASIA-Webface', 'r100', '/home/bjgbiesseck/GitHub/insightface/recognition/arcface_torch/work_dirs/casia_frcsyn_r100/2023-10-14_09-51-11_GPU0/model.pt')]



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='summary', help='\'summary\' or \'inference\'')
    args = parser.parse_args()
    return args


def load_text_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def save_list_as_csv_table(data, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)


def parse_stderr_output(lines, target, face_attribs):
    general_results = True
    results = {}
    if lines is str:
        lines = lines.strip().split('\n')
    for line in lines:
        # print('line:', line)
        line_split = line.split(target)[-1]
        line_split = line_split.lstrip(']')
        line_split = [line.strip() for line in line_split.split('  ')]
        line_split = [line for line in line_split if len(line) > 0]
        line_split = [line.split(': ') for line in line_split]
        # print('line_split:', line_split)

        for pair in line_split:
            if '----' in pair[0]:
                general_results = False
                continue
            # print('pair:', pair, '    general_results:', general_results)

            if general_results:
                # results[pair[0]] = pair[1]
                results[pair[0].split('=')[0]] = float(pair[1].split('+-')[0])
            else:
                face_attrib, atribvalue_and_metric = pair[0][1:].split(']')
                # idx_whitespace = atribvalue_and_metric.find(' ')
                idxs_whitespace = [i for i, c in enumerate(atribvalue_and_metric) if c == ' ']
                group = atribvalue_and_metric[:idxs_whitespace[-1]]
                metric = atribvalue_and_metric[idxs_whitespace[-1]+1:].split('=')[0]
                metric_value = pair[1]
                # print(f'face_attrib: \'{face_attrib}\'    group: \'{group}\'    metric: \'{metric}\'    metric_value: \'{metric_value}\'')
                
                if not face_attrib in results: results[face_attrib] = {}
                if not metric in results[face_attrib]: results[face_attrib][metric] = {}
                if not group in results[face_attrib][metric]: results[face_attrib][metric][group] = {}
                
                # results[face_attrib][metric][group] = metric_value
                results[face_attrib][metric][group] = float(metric_value.split('+-')[0])

            # print('#######')

    return results



args = parse_arguments()
face_attribs = ['race', 'gender', 'age']
results = {}
for idx_tgt, (target, arguments) in enumerate(benchmarks):
    results[target] = {}
    for idx_model, (model_name, network, model_path) in enumerate(models):
        print(f'target={idx_tgt}/{len(benchmarks)}, model={idx_model}/{len(models)} - target: {target}, model_name: {model_name}')
            
        if args.type == 'summary':

            path_dir_model = os.path.join(os.path.dirname(model_path), f'eval_{target.lower()}')
            path_file_results = os.path.join(path_dir_model, 'results_logs')
            if '--protocol' in arguments:
                arguments_split = arguments.split(' ')
                path_file_results += f"_prot={os.path.basename(arguments_split[arguments_split.index('--protocol')+1])}"
            path_file_results += '.txt'
            print('path_file_results:', path_file_results, '    exists:', os.path.isfile(path_file_results))

            if os.path.isfile(path_file_results):
                results_lines = load_text_file(path_file_results)
                # print('results_lines:', results_lines)

                parsed_results = parse_stderr_output(results_lines, target, face_attribs)
                print('parsed_results:')
                print(parsed_results)

                parsed_results_json = json.dumps(parsed_results, indent=4)
                # print('parsed_results_json:')
                # print(parsed_results_json)
                # print('type(parsed_results_json):', type(parsed_results_json))

            results[target][model_name] = parsed_results


        elif args.type == 'inference':
            arguments_final = arguments % (network, model_path)
            cmd_final = ['python', 'verification_TCDiff.py']
            cmd_final.extend(arguments_final.split(' '))
            
            # print(f'target={idx_tgt}/{len(benchmarks)}, model={idx_model}/{len(models)} - target: {target}, model_name: {model_name}')
            result = subprocess.run(cmd_final, capture_output=True, text=True)

            stdout    = result.stdout
            stderr    = result.stderr
            exit_code = result.returncode

            # stdout = stdout.split('\n')
            # stderr = stderr.split('\n')

            # print('stdout:', stdout)
            print('stderr:\n', stderr)
            print('exit_code:', exit_code)
            # sys.exit(0)

            # results[target][model_name] = {'stdout': stdout, 'stderr': stderr, 'exit_code': exit_code}
            results[target][model_name] = {'stderr': stderr}

            # parsed_results = parse_stderr_output(results[target][model_name]['stderr'], target)
            # print('parsed_results:')
            # print(parsed_results)

            print('-----')
    print('===================')



# print('results:')
# print(results)
# results_json = json.dumps(results, indent=4)
# results_json = json.dumps(results)
# print('results_json:')
# print(results_json)
# print('\n')
# with open("results.json", "w") as outfile:
#     outfile.write(results_json)
# sys.exit(0)



'''
data = [
    {"Name": "Alice",   "Age": 30, "City": "New York"},
    {"Name": "Bob",     "Age": 25, "City": "Los Angeles"},
    {"Name": "Charlie", "Age": 35, "City": "Chicago"},
    {"Name": "Dave",    "Age": 28, "City": "Houston"}
]

data = [
    {"Model": "R100/CASIA-Webface",    "hda_doppelganger": XX,    "doppelver_doppelganger": XX,    "doppelver_vise": XX,    "3d_tec": XX},
    {"Model": "R100/MS1MV3",           "hda_doppelganger": XX,    "doppelver_doppelganger": XX,    "doppelver_vise": XX,    "3d_tec": XX},
    {"Model": "R100/Glint360K",        "hda_doppelganger": XX,    "doppelver_doppelganger": XX,    "doppelver_vise": XX,    "3d_tec": XX},
    {"Model": "R100/DCFace",           "hda_doppelganger": XX,    "doppelver_doppelganger": XX,    "doppelver_vise": XX,    "3d_tec": XX},
]
'''
results_formatted = {}
for idx_key, key in enumerate(results[benchmarks[0][0]][models[0][0]]):
    results_key = []
    for idx_model, (model_name, network, model_path) in enumerate(models):
        results_model = {'Model': model_name}
        for idx_tgt, (target, arguments) in enumerate(benchmarks):
            print(f'idx_key={idx_key}/{len(results[benchmarks[0][0]][models[0][0]])}, idx_model={idx_model}/{len(models)}, idx_tgt={idx_tgt}/{len(benchmarks)}', '    key:', key, '    model_name:', model_name, '    target:', target, '    item:', results[target][model_name][key])

            if not isinstance(results[target][model_name][key], dict): # general results
                # print('    general')
                results_model[target] = results[target][model_name][key]

            else:  # face attribs
                # print('    face attrib')
                results_model[target] = results[target][model_name][key]

        results_key.append(results_model)
    print('---')
    results_formatted[key] = results_key
            

# print('results_formatted')
# print(results_formatted)

# results_formatted_json = json.dumps(results_formatted, indent=4)
# print('results_formatted_json:')
# print(results_formatted_json)

# print('results:')
# print(results)

# sys.exit(0)

metrics_to_save = ['Accuracy-Flip', 'TAR@FAR', 'EER', 'race', 'gender', 'age']
for idx_metric, metric in enumerate(metrics_to_save):
    output_file_name = f'{metric}.csv'
    print(f'Saving table \'{output_file_name}\'')
    save_list_as_csv_table(results_formatted[metric], output_file_name)


print('\nFinished!\n')