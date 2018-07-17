#!/bin/bash

# Copy to the directory containing 'SingleCaptureImages' and 'RecapturedImages' directories
# 	prior to running this script.

ROOT_PATH=`pwd`

cd $ROOT_PATH/SingleCaptureImages

cd EOS600D
for filename in `ls | grep -v EOS`
do
	mv $filename ${filename//600D/EOS600D}
done

cd $ROOT_PATH/RecapturedImages

mv D70s D70S
mv 60D EOS60D
mv 600D EOS600D
