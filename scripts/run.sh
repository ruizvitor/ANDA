wget http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip
unzip MSRA10K_Imgs_GT.zip

rm Readme.txt

bash protocol.sh MSRA10K_Imgs_GT/Imgs

ORIGINALPATH=$(pwd)

BG_PATH=$ORIGINALPATH/../PConvInpainting/output/

cd ../PConvInpainting
mkdir output
python inpaintMSRA10K.py --img_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/images/ --mask_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/masks/ --out_path $BG_PATH
cd ../scripts/featureRelated

python computeKnn.py --obj_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/images/ --obj_mask_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/masks/ --bg_path $BG_PATH
mkdir -p output/images
mkdir -p output/masks
python anda.py --obj_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/images/ --obj_mask_path $ORIGINALPATH/MSRA10K_Imgs_GT/Imgs/masks/ --bg_path $BG_PATH


cd $ORIGINALPATH
