import argparse
arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


main_arg = add_argument_group('Main')


main_arg.add_argument('--mode', type=str,
                      default='Training',
                      help='Running mode')

main_arg.add_argument('--save_dir', type=str,
                      default='./',
                      help='Save model weights')

main_arg.add_argument('--log_dir', type=str,
                      default='/projects/ovcare/classification/ywang/project_log/midl_768_monoscale_log/',
                      help='TensorBoard directory')

main_arg.add_argument('--dataset_dir', type=str,
                      default='/projects/ovcare/classification/ywang/midl_dataset/768_monoscale/',
                      help='Slide and patch id files directory')

main_arg.add_argument('--train_ids_file_name', type=str,
                      default='patch_ids/1_2_train_3_eval_train_ids.txt',
                      help='Training patch path ids')

main_arg.add_argument('--val_ids_file_name', type=str,
                      default='patch_ids/1_2_train_3_eval_eval_0_ids.txt',
                      help='Validation patch path ids')

main_arg.add_argument('--test_ids_file_name', type=str,
                      default='patch_ids/1_2_train_3_eval_eval_1_ids.txt',
                      help='Testing patch path ids')

main_arg.add_argument('--testing_output_file_name',
                      type=str, default='a_patch_probability.txt')

main_arg.add_argument('--preload_image_file_name',
                      type=str, default='768_monoscale.h5')

main_arg.add_argument('--count_fusion_classifier', type=str,
                      default='RandomForest')

main_arg.add_argument('--count_exclude_mode', type=str,
                      default='gap')

main_arg.add_argument('--count_exclude_threshold', type=float,
                      default=0.8)

main_arg.add_argument('--model_name_prefix', type=str, default='')

main_arg.add_argument('--epoch', type=int,
                      default=20,
                      help='Number of epoches')

main_arg.add_argument('--batch_size', type=int,
                      default=64,
                      help='Batch size')

main_arg.add_argument('--lr', type=float,
                      default=0.0002,
                      help='Learning rate')

main_arg.add_argument('--rep_intv', type=int,
                      default=250,
                      help='Report interval')

main_arg.add_argument('--n_eval_samples', type=int,
                      default=2000,
                      help='Number of samples for eval during training')

main_arg.add_argument('--n_subtypes', type=int,
                      default=5)

main_arg.add_argument('--expert_magnification', type=str,
                      default=256)

main_arg.add_argument('--optim', type=str, default='Adam')

main_arg.add_argument('--deep_model', type=str, default='DeepModel')

main_arg.add_argument('--deep_classifier', type=str, default='baseline')

main_arg.add_argument('--continue_train', action='store_true')

main_arg.add_argument('--is_multiscale_expert', action='store_true')

main_arg.add_argument('--use_pretrained', action='store_true')

main_arg.add_argument('--use_equalized_batch', action='store_true')

main_arg.add_argument('--load_model_id', type=str,
                      default='max_val_acc')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
