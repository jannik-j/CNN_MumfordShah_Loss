python Axon_train_unet.py           ^
--dataroot F:\Daten_Bachelorarbeit\Axon ^
--gpu_ids 0                 ^
--name Axon_subpixelUnet_Adam    ^
--model unet       ^
--which_model_netG subpixelUnet ^
--init_type xavier ^
--optim Adam ^
--no_dropout ^
--norm batch ^
--batchSize 8 ^
--semi_rate 10  ^
--lr 0.01 ^
--lr_policy step ^
--lr_decay_iters 100 ^
--inputSize 512 ^
--fineSize 256 ^
--input_nc 1 ^
--output_nc 3 ^
--niter 600 ^
--display_step 5  ^
--plot_step 5             ^
--save_epoch_freq 100        ^
--display_port 8097         ^
--segType tem ^
--lambda_A 0.0000001 ^
--checkpoints_dir ./checkpoints/2020-07-29_Axon_semi_beta=10-7_semi=10  ^
--niter_decay 0 ^
--no_html
