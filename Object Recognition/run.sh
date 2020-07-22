rm -rf ./txt &
wait
mkdir txt &
wait
rm -rf ./graph &
wait
rm -rf ./incorrect &
wait
rm -f discriminative_dann.pyc &
wait

nohup python office.py --source_set 'amazon' --target_set 'webcam' --gpu '0' --batch_size 16 --base_lr 0.0015\
                       --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/a2w.txt &

# nohup python office.py --source_set 'amazon' --target_set 'dslr' --gpu '1' --batch_size 16 --base_lr 0.0015\
#                        --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/a2d.txt &

# nohup python office.py --source_set 'webcam' --target_set 'amazon' --gpu '2' --batch_size 16 --base_lr 0.0015\
#                        --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/w2a.txt &

# nohup python office.py --source_set 'dslr' --target_set 'amazon' --gpu '0' --batch_size 16 --base_lr 0.0015\
#                        --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/d2a.txt &

# nohup python office.py --source_set 'dslr' --target_set 'webcam' --gpu '1' --batch_size 16 --base_lr 0.0015\
#                        --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/d2w.txt &

# nohup python office.py --source_set 'webcam' --target_set 'dslr' --gpu '0' --batch_size 16 --base_lr 0.0015\
#                        --pretrain_sample 50000 --train_sample 200000 --form_w 0.4 --main_w -0.8 --nesterov True > ./txt/w2d.txt &

# nohup python clef.py --source_set 'i' --target_set 'p' --gpu '1' --batch_size 8 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8 --wp 0.1 > ./txt/i2p.txt &

# nohup python clef.py --source_set 'p' --target_set 'c' --gpu '1' --batch_size 8 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8 --wp 0.1 > ./txt/p2i.txt &

# nohup python clef.py --source_set 'c' --target_set 'p' --gpu '0' --batch_size 16 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8  > ./txt/c2p.txt &

# nohup python clef.py --source_set 'p' --target_set 'c' --gpu '0' --batch_size 16 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8  > ./txt/p2c.txt &

# nohup python clef.py --source_set 'i' --target_set 'c' --gpu '2' --batch_size 16 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8  > ./txt/i2c.txt &

# nohup python clef.py --source_set 'c' --target_set 'i' --gpu '2' --batch_size 16 --base_lr 0.001\
#                        --pretrain_sample 30000 --train_sample 120000 --form_w 0.4 --main_w -0.8  > ./txt/c2i.txt &
