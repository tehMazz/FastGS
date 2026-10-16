SCENES=(
  COGEN1
  COGEN2
  Compressor
  G35
  PCV
  TA_BLUE_MOTOR
  TA_TURBINE
)

SCENES=(
  TTA
)

for scene in "${SCENES[@]}"; do
  job_id="${scene,,}_bakerh"
  CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID="$job_id" python train.py -s "./datasets/BakerH_undis/$scene" -i images --images images_1600 --eval --densification_interval 100 --optimizer_type default --test_iterations 30000 --grad_abs_thresh 0.001 --data_device cpu
done

for scene in "${SCENES[@]}"; do
  job_id="${scene,,}_bakerh"
  CUDA_VISIBLE_DEVICES=1 python render.py -m "output/$job_id" --skip_train
done

for scene in "${SCENES[@]}"; do
  job_id="${scene,,}_bakerh"
  CUDA_VISIBLE_DEVICES=1 python metrics.py -m "output/$job_id"
done