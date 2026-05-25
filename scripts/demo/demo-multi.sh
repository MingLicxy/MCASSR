# in-scale
echo '#######################[x2]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 96,96 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x2.png --gpu $1 &&
echo '#######################[x3]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 144,144 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x3.png --gpu $1 &&
echo '#######################[x4]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 192,192 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x4.png --gpu $1 &&
echo '#######################[x6]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 288,288 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x6.png --gpu $1 &&
echo '#######################[x8]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 384,384 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x8.png --gpu $1 &&
echo '#######################[x12]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 576,576 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x12.png --gpu $1 &&
echo '#######################[x18]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 864,864 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x18.png --gpu $1 &&
echo '#######################[x24]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 1152,1152 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x24.png --gpu $1 &&
echo '#######################[x30]#######################' &&
python /home/caoxinyu/Arbitrary-scale/liif-main/demo.py --input /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/LR/29.png --model /home/caoxinyu/Arbitrary-scale/liif-main/save/_train_mamba_cnn_1-ciaosr_liif_full_4444/epoch-best.pth --resolution 1440,1440 --output /home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/SR/x30.png --gpu $1 &&
true