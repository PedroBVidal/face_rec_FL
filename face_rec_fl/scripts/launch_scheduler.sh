#!/bin/bash
# Launcher — fully detaches the scheduler from any parent shell
cd /home/pedro.vidal/facerec_flower/face_rec_fl
exec nohup bash /home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/scripts/scheduler_emore.sh \
    > /home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs/scheduler_logs/nohup_v4.out 2>&1 &
