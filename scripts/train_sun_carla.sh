python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
            ../train_sun_model.py \
            --ngpu=1 \
            --dataset=lhq \
            --height=256 \
            --width=256 \
            --stereo_baseline=0.54 \
            --batch_size=1 \
            --num_epochs=30 \
            --mode=train \
            --num_classes=183 \
            --embedding_size=183 \
            --port=6071 \
            --data_path=/root/gvsnet/datasets/lhq2/LHQwarp \
            --logging_path=/root/gvsnet/datasets/lhq2/sun \
            --image_log_interval=2000 \

