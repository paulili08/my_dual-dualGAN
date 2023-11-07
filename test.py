import os
cmd = 'python main.py --phase test --dataset_name eeg2audio --participant sub-01 --image_size 128 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100'
os.system(cmd)
cmd = 'python main2.py --phase test --dataset_name eeg2audio --participant sub-01 --image_size 128 --lambda_B 1000.0 --lambda_C 1000.0 --epoch 100'
os.system(cmd)
