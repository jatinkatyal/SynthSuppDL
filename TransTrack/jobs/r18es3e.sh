#!/bin/bash
#SBATCH --time=7:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --account=def-cpoullis
#SBATCH --mem=32000M
#SBATCH --gres=gpu:4
#SBATCH --output="/home/jatin/scratch/slurm-%j.out"
#SBATCH --mail-user=jatinkatyal96@gmail.com
#SBATCH --mail-type=ALL

module load opencv python/3.7 cuda/11.4 cudnn
source /home/jatin/transtrack/bin/activate

cd /home/jatin/work/tracking_analysis/TransTrack
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main_track.py  --output_dir /home/jatin/projects/def-cpoullis/jatin/TransTrack/r18es3e --dataset_file r18es3e --coco_path mixsets/r18es3e --batch_size 4 --resume output_crowdhuman/crowdhuman_final.pth --with_box_refine --num_queries 500  --epochs 20 --lr_drop 10
