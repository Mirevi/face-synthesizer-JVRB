cd ../
echo "./.venv/Scripts/python.exe" -m train --name 0_0_p2pG_L1_only --model_type l1 
echo "./.venv/Scripts/python.exe" -m train --name 1_0_p2p_512 --model_type pix2pix 
echo "./.venv/Scripts/python.exe" -m train --name 1_1_p2p_512_lsgan --model_type pix2pix --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 1_2_p2p_512_Mapping --use_mapping_network --netM mlp_with_conv_512 
echo "./.venv/Scripts/python.exe" -m train --name 1_3_p2p_512_Mapping_lsgan --use_mapping_network --netM mlp_with_conv_512 --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 1_4_p2p_512_Mapping --use_mapping_network --netM residual 
echo "./.venv/Scripts/python.exe" -m train --name 1_5_p2p_512_Mapping_lsgan --use_mapping_network --netM residual --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 1_6_p2p_512_Mapping --use_mapping_network --netM mlp_and_residual_512 
echo "./.venv/Scripts/python.exe" -m train --name 1_7_p2p_512_Mapping_lsgan --use_mapping_network --netM mlp_and_residual_512 --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 2_0_p2phd_feat_loss --use_gan_feat_loss 
echo "./.venv/Scripts/python.exe" -m train --name 2_1_p2phd_feat_loss_lsgan --use_gan_feat_loss --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 3_0_lpips_loss --use_lpips_loss 
echo "./.venv/Scripts/python.exe" -m train --name 3_1_lpips_loss_lsgan --use_lpips_loss --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 4_0_cycle_loss --use_cycle_loss 
echo "./.venv/Scripts/python.exe" -m train --name 4_1_cycle_loss_lsgan --use_cycle_loss --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 5_0_multiscale_D --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 5_1_multiscale_D_lsgan --use_multiscale_D --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 5_2_multiscale_D_no_L1 --no_L1_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 5_3_multiscale_D_no_L1_lsgan --no_L1_loss --use_multiscale_D --gan_mode lsgan 
echo "./.venv/Scripts/python.exe" -m train --name 6_0_all --gan_mode lsgan --use_gan_feat_loss --use_lpips_loss --use_cycle_loss --no_L1_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 6_1_all_lpips_10 --gan_mode lsgan --use_gan_feat_loss --use_lpips_loss --lambda_lpips 10 --use_cycle_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 6_2_feat_multiD --gan_mode lsgan --use_gan_feat_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 6_3_feat_multiD_noL1 --gan_mode lsgan --use_gan_feat_loss --use_multiscale_D --no_L1_loss 
echo "./.venv/Scripts/python.exe" -m train --name 6_4_feat_multiD_lpips_10 --gan_mode lsgan --use_lpips_loss --lambda_lpips 10 --use_gan_feat_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 6_5_feat_multiD_lpips_10_cycle --gan_mode lsgan --use_lpips_loss --lambda_lpips 10 --use_gan_feat_loss --use_multiscale_D --use_cycle_loss 
echo "./.venv/Scripts/python.exe" -m train --name 7_0_epochs_30_70 --n_epochs 30 --n_epochs_decay 70 --gan_mode lsgan --use_lpips_loss --lambda_lpips 10 --use_gan_feat_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 7_1_epochs_30_30 --n_epochs 30 --n_epochs_decay 30 --gan_mode lsgan --use_lpips_loss --lambda_lpips 10 --use_gan_feat_loss --use_multiscale_D 
echo "./.venv/Scripts/python.exe" -m train --name 8_0_epochs_30_70 --n_epochs 30 --n_epochs_decay 70 --gan_mode lsgan --use_lpips_loss --lambda_lpips 10 --use_gan_feat_loss --use_multiscale_D --dataset_name philipp_train --eval_dataset_name philipp_eval 

echo "./.venv/Scripts/python.exe" -m train --dataset_name Kester_18-02-22 --name Kester_no_cycle_loss_with_pca --n_epochs 30 --n_epochs_decay 70 --netD multiscale --use_gan_feat_loss --use_lpips_loss --lambda_lpips 10 
echo "./.venv/Scripts/python.exe" -m train --dataset_name Jannik_21-02-22 --name Jannik_no_cycle_loss_with_pca --n_epochs 30 --n_epochs_decay 70 --netD multiscale --use_gan_feat_loss --use_lpips_loss --lambda_lpips 10 
"./.venv/Scripts/python.exe" -m train --dataset_name Philipp_24-02-22 --name Philipp_no_cycle_loss_with_pca --n_epochs 30 --n_epochs_decay 70 --netD multiscale --use_gan_feat_loss --use_lpips_loss --lambda_lpips 10 
cd scripts

echo shutdown -s

echo --continue_train --epoch_count 30 --epoch 95