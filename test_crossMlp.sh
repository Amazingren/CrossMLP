python test.py --dataroot /datasets/path/of/the/dataset \
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

# python test.py --dataroot ./datasets/dayton_a2g \
# --name dayton_a2g_selectiongan \
# --model selectiongan \
# --which_model_netG unet_256 \
# --which_direction AtoB \
# --dataset_mode aligned \
# --norm batch \
# --gpu_ids 0 \
# --batchSize 4 \
# --loadSize 286 --fineSize 256 --no_flip --eval
