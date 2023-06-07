#!/usr/bin/bash
# Activate conda environment
source /home/pkdadmin/miniconda3/etc/profile.d/conda.sh && conda activate nwc
# Go to opera data directory
cd /mnt/Source_data/home/lb/opera
# Find latest file and extract datetime from its name
datetime=$(find -type f | sort -r | head -n 1 | cut -d "_" -f 5 | cut -d "." -f 1)
# Go to nowcasting script directory
cd /home/pkdadmin/products/nwp/dp/orkans
# Run nowcast
/home/pkdadmin/miniconda3/envs/nwc/bin/python ./main.py --datetime $datetime
