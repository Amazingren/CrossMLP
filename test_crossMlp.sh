python test.py --dataroot [path_to_dataset] \
--name crossMlp_dayton_ablation \
--model crossmlpgan \
--which_model_netG unet_256 \
--which_direction AtoB \
--dataset_mode aligned \
--norm batch \
--gpu_ids 0 \
--batchSize 8 \
--loadSize 286 \
--fineSize 256 \
--saveDisk  \ 
--no_flip --eval
