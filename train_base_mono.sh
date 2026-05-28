CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/bicycle -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --grad_abs_thresh 0.0008
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/flowers -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --dense 0.005 --grad_abs_thresh 0.001
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/garden -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0003
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=stump_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/stump -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.004 --grad_abs_thresh 0.001
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=treehill_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/treehill -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.01 --grad_abs_thresh 0.0018
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=room_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/room -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0004 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/counter -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0004
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/kitchen -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai_big python train.py -s /leonardo_work/IscrC_EditGS/datasets/mip360/bonsai -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002

CUDA_VISIBLE_DEVICES=0 python render.py -m output/bicycle_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/flowers_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/garden_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/stump_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/treehill_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/room_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/counter_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/bonsai_big --skip_train