ORIGINALPATH=$(pwd)
cd $1

mkdir images
mkdir masks

mv *.jpg images/
mv *.png masks/

cd images

COUNT=0
for i in `ls -v *.jpg`;
do

mv $i $(printf "MSRA10K_image_%06d.jpg" "$COUNT")
COUNT=$((COUNT+1))

done

cd ..

cd masks

COUNT=0
for i in `ls -v *.png`;
do

mv $i $(printf "MSRA10K_mask_%06d.png" "$COUNT")
COUNT=$((COUNT+1))

done

cd ..

cd $ORIGINALPATH
