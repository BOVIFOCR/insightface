from easydict import EasyDict as edict
import os
uname = os.uname()

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
# config.batch_size = 128
# config.batch_size = 64
config.batch_size = 16

# config.lr = 0.1
# config.lr = 0.05
config.lr = 0.01
# config.lr = 0.0075
# config.lr = 0.005
# config.lr = 0.001

config.verbose = 2000
# config.verbose = 10
config.dali = False


config.loss = 'CombinedMarginLoss'   # default
# config.loss = 'LDAMLoss'
# config.loss = 'FocalLoss'

config.train_rule = None             # default
# config.train_rule = 'Resample'
# config.train_rule = 'Reweight'
# config.train_rule = 'DRW'



if uname.nodename == 'duo':
    # config.rec = "/train_tmp/faces_emore"

    # λ_BFM = 0.01
    # config.rec = '/datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100'
    # config.rec = '/datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100'
    config.rec = '/datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_FROM-BUPT_NCLUSTERS=100'

    # λ_BFM = 0.005
    # config.rec = ''

    # λ_BFM = 0.001
    # config.rec = ''

    # config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
    config.val_targets = ['/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/lfw.bin',
                          '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cplfw.bin',
                          '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_ff.bin',
                          '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_fp.bin',
                          '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/agedb_30.bin',
                          '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/calfw.bin',
                          'bupt']
    # config.val_targets = ['bupt']
    config.val_dataset_dir = ['/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
    config.val_protocol_path = ['/datasets2/1st_frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']


elif uname.nodename == 'diolkos':
    # λ_BFM = 0.01
    # config.rec = '/nobackup/unico/datasets/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100'
    # config.rec = '/nobackup/unico/datasets/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100'
    config.rec = '/nobackup/unico/datasets/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_FROM-BUPT_NCLUSTERS=100'

    # λ_BFM = 0.005
    # config.rec = ''

    # λ_BFM = 0.001
    # config.rec = ''

    config.val_targets = ['/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/lfw.bin',
                          '/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cplfw.bin',
                          '/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_ff.bin',
                          '/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_fp.bin',
                          '/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/agedb_30.bin',
                          '/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/calfw.bin',
                          'bupt']
    # config.val_targets = ['bupt']
    config.val_dataset_dir = ['/nobackup/unico/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
    config.val_protocol_path = ['/nobackup/unico/1st_frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']


elif uname.nodename == 'daugman':

    # λ_BFM = 0.01
    # config.rec = ''

    # λ_BFM = 0.005
    # config.rec = ''
    
    # λ_BFM = 0.001
    # config.rec = ''
    
    # config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
    config.val_targets = ['/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/lfw.bin',
                          '/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cplfw.bin',
                          '/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_ff.bin',
                          '/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/cfp_fp.bin',
                          '/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/agedb_30.bin',
                          '/groups/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112/calfw.bin',
                          'bupt']
    # config.val_targets = ['bupt']
    config.val_dataset_dir = ['/groups/unico/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
    config.val_protocol_path = ['/groups/unico/1st_frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']


else:
    raise Exception(f'Paths of train and val datasets could not be found in file \'{__file__}\'')


# config.num_classes = 85742
config.num_classes = 10000

# config.num_image = 5822653
config.num_image = 500000

config.num_epoch = 20
# config.num_epoch = 30
config.warmup_epoch = 0



# WandB Logger
# config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.wandb_key = "d7664714c72bd594a957381812da450279f80f66"

config.suffix_run_name = None

# config.using_wandb = False
config.using_wandb = True

# config.wandb_entity = "entity"
# config.wandb_entity = "dcface_fr"
config.wandb_entity = 'biesseck'

# config.wandb_project = "R100_DCFace_local_trained_default_1000subj"
config.wandb_project = "tcdiff_fr_sty-equal_100clusters_10000subj"
config.wandb_log_all = True

# config.save_artifacts = False
config.save_artifacts = True

config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted

config.notes = ''
