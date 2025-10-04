export EXTERNAL_DIR=./external

cp update_src/code/rdd.py $EXTERNAL_DIR/hloc/hloc/extractors/rdd.py
cp update_src/code/aliked.py $EXTERNAL_DIR/hloc/hloc/extractors/aliked.py
cp update_src/code/extract_features.py $EXTERNAL_DIR/hloc/hloc/extract_features.py

bash update_src/setup_rdd.sh

echo "=== Update src code successfully! ==="