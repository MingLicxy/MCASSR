# in-scale
# echo '#######################[div2k-x2]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-2.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x3]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-3.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x4]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-4.yaml --model $1 --gpu $2 &&

# out-of-scale
# echo '#######################[div2k-x6*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-6.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x8*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-8.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x12*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-12.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x18*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-18.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x24*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-24.yaml --model $1 --gpu $2 &&
# echo '#######################[div2k-x30*]#######################' &&
# python /home/caoxinyu/Arbitrary-scale/liif-main/test.py --config /home/caoxinyu/Arbitrary-scale/liif-main/configs/test-liif/test-Flir/test-div2k-30.yaml --model $1 --gpu $2 &&

# 非整数尺度
echo '#######################[div2k-x2]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/DLS-NUC-100/bicubic/X2/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/DLS-NUC-100_light_x2 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 2 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/IR100/bicubic/X2/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/IR100_light_x2 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 2 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-A/bicubic/X2/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-A_light_x2 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 2 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-C/bicubic/X2/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-C_light_x2 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 2 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/Flir/bicubic/X2/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/Flir_light_x2 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 2 --gpu 1 &&

echo '#######################[div2k-x3]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/DLS-NUC-100/bicubic/X3/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/DLS-NUC-100_light_x3 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 3 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/IR100/bicubic/X3/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/IR100_light_x3 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 3 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-A/bicubic/X3/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-A_light_x3 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 3 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-C/bicubic/X3/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-C_light_x3 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 3 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/Flir/bicubic/X3/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/Flir_light_x3 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 3 --gpu 1 &&

echo '#######################[div2k-x4]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/DLS-NUC-100/bicubic/X4/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/DLS-NUC-100_light_x4 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 4 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/IR100/bicubic/X4/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/IR100_light_x4 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 4 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-A/bicubic/X4/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-A_light_x4 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 4   --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/results-C/bicubic/X4/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/results-C_light_x4 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 4 --gpu 1 &&

python /home/caoxinyu/Arbitrary-scale/liif-main/demo_new.py --input_dir /home/caoxinyu/Arbitrary-scale/data/test_data/Flir/bicubic/X4/LR --output_dir /home/caoxinyu/Arbitrary-scale/liif-main/results/Flir_light_x4 --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_2222/epoch-best.pth --scale 4 --gpu 1 &&

true
