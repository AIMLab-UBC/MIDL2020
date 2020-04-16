#!/bin/bash
chmod 775 ./patch_level.py
echo 'Two-stage model using split A'

# echo 'Stage 1 - Patch Size 256 * 256 Training'
# ./patch_level.py  --deep_model DeepModel --deep_classifier two_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 64 --epoch 1 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/test_dataset --preload_image_file_name dataset.h5 --train_ids_file_name patch_ids/patch_ids.txt  --val_ids_file_name patch_ids/patch_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/test_dataset_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/
# echo 'Stage 1 - Patch Size 256 * 256 Validation'
# ./patch_level.py  --mode Validation --deep_model DeepModel --deep_classifier two_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 64 --epoch 1 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/test_dataset --preload_image_file_name dataset.h5 --train_ids_file_name patch_ids/patch_ids.txt  --val_ids_file_name patch_ids/patch_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/test_dataset_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/

# echo 'Stage 2 - Patch Size 512 * 512 Training'
# ./patch_level.py  --deep_model DeepModel --deep_classifier two_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 32 --epoch 1 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/test_dataset --preload_image_file_name dataset.h5 --train_ids_file_name patch_ids/patch_ids.txt  --val_ids_file_name patch_ids/patch_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/test_dataset_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/
# echo 'Stage 2 - Patch Size 512 * 512 Validation'
# ./patch_level.py  --mode Validation --deep_model DeepModel --deep_classifier two_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 32 --epoch 1 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/test_dataset --preload_image_file_name dataset.h5 --train_ids_file_name patch_ids/patch_ids.txt  --val_ids_file_name patch_ids/patch_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/test_dataset_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/

#!/bin/bash
# echo 'Maestro v2 Model Split A'

echo 'Progressive Resizing Stage 1 - Patch Size 256 * 256'
#./maestro.py  --deep_model Maestro --maestro_version MaestroV2 --model_name_prefix a_maestrov2 --use_pretrained --lr 0.0002 --batch_size 64 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_progressive --preload_image_file_name 1024_progressive_384.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt  --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/maestro_log/ --save_dir /projects/ovcare/classification/ywang/project_save/maestro_save/
./patch_level.py  --deep_model DeepModel --mode Validation --model_name_prefix a_maestrov2 --use_pretrained --lr 0.0002 --batch_size 64 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_progressive --preload_image_file_name 1024_progressive_384.h5 --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --test_ids_file_name patch_ids/1_2_train_3_eval_eval_1_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/maestro_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/

#echo 'Progressive Resizing Stage 2 - Patch Size 512 * 512'
#./maestro.py  --deep_model Maestro --maestro_version MaestroV2 --model_name_prefix a_maestrov2 --use_pretrained --lr 0.0002 --batch_size 32 --epoch 20 --rep_intv 500 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_progressive --preload_image_file_name 1024_progressive_384.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/maestro_log/ --save_dir /projects/ovcare/classification/ywang/project_save/maestro_save/
./patch_level.py  --deep_model DeepModel --mode Validation --model_name_prefix a_maestrov2 --use_pretrained --lr 0.0002 --batch_size 32 --epoch 20 --rep_intv 500 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_progressive --preload_image_file_name 1024_progressive_384.h5 --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --test_ids_file_name patch_ids/1_2_train_3_eval_eval_1_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/maestro_log/ --save_dir /projects/ovcare/classification/ywang/project_save/test_dataset_save/