python test.py configs/efficientnetv2-b3_8xb16_stpt.py\
        work_dirs/STPT/efficientnetv2_b3/best_accuracy_top1_epoch_96.pth\
        --work-dir results/STPT/efficientnetv2_b3\
        --out results/STPT/efficientnetv2_b3/result.json\
        --out-item metrics\
        --gsheets