import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='inference')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./result/checkpoint')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./result/output/')
    parser.add_argument(
        '--bpm_batch_size',
        type=int,
        default='16')
    parser.add_argument(
        '--abi_batch_size',
        type=int,
        default='16')
    parser.add_argument(
        '--bpm_check_num',
        type=int,
        default='10')
    parser.add_argument(
        '--abi_check_num',
        type=int,
        default='10')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_bpm_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--train_abi_epochs',
        type=int,
        default=10)
    parser.add_argument(
        '--step_size',
        type=int,
        default=10)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5)

    parser.add_argument(
        '--proposal_thres',
        type=float,
        default=0.7)

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/thumos_annotations/")
    
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=256)
    parser.add_argument(
        '--max_D',
        type=int,
        default=64)
    parser.add_argument(
        '--min_D',
        type=int,
        default=0)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="./data/thumos_feature/")


    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=70)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./result/output/result_proposal.json")
    parser.add_argument(
        '--gama',
        type=float,
        default=21)
    args = parser.parse_args()

    return args

