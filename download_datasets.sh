mkdir -p data

#echo "Downloading codrna from LIBSVM"
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna -O data/codrna.train
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t -O data/codrna.test
#python python-utils/shuffle.py data/codrna.train -s 0 -if libsvm -of csv
#python python-utils/scale_y.py data/codrna.train -if csv -of csv
#python python-utils/scale_y.py data/codrna.test -if libsvm -of csv
#
#echo "Downloading cadata from LIBSVM"
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata -O data/cadata
#python python-utils/scale_y.py data/cadata -if libsvm -of csv
#python python-utils/train_test_split.py -v 0 -t 0.2 -s 0 --scale-y data/cadata -if csv -of csv
#
#echo "Downloading covtype from LIBSVM"
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 -O data/covtype.bz2
#bzip2 -d data/covtype.bz2
#python python-utils/scale_y.py data/covtype -if libsvm -of csv
#python python-utils/train_test_split.py -v 0 -t 0.2 -s 0 --scale-y data/covtype -if csv -of csv
#
#echo "Downloading gisette from LIBSVM"
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2 -O data/gisette.train.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2 -O data/gisette.test.bz2
#bzip2 -d data/gisette.train.bz2
#bzip2 -d data/gisette.test.bz2
#python python-utils/shuffle.py data/gisette.train -s 0 -if libsvm -of csv
#python python-utils/scale_y.py data/gisette.train -if csv -of csv
#python python-utils/scale_y.py data/gisette.test -if libsvm -of csv
#
#
#echo "Downloading msd from LIBSVM"
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2 -O data/msd.train.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2 -O data/msd.test.bz2
#bzip2 -d data/msd.train.bz2
#bzip2 -d data/msd.test.bz2
#python python-utils/shuffle.py data/msd.train -s 0 -if libsvm -of csv
#python python-utils/scale_year_msd.py data/msd.train -if csv -of csv
#python python-utils/scale_year_msd.py data/msd.test -if libsvm -of csv

#echo "Removing samples from datasets"
#for dataset in codrna cadata covtype gisette msd; do
#  python python-utils/remove_sample.py -s 0 -r 0.01 data/"$dataset".train -if csv -of csv
#  python python-utils/remove_sample.py -s 0 -r 0.001 data/"$dataset".train -if csv -of csv
#done

#echo "Downloading susy from UCI Machine Learning Repository"
#wget https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.gz -O data/susy.gz
#gzip -d data/susy.gz
#python python-utils/scale_y.py data/susy -if csv -of csv
#python python-utils/train_test_split.py -v 0 -t 0.2 -s 0 --scale-y data/susy -if csv -of csv

echo "Removing samples from datasets"
for dataset in codrna cadata covtype gisette msd; do
  python python-utils/remove_sample.py -s 0 -r 0.05 -if csv -of csv data/"$dataset".train
  python python-utils/remove_sample.py -s 0 -r 0.1 -if csv -of csv data/"$dataset".train
  python python-utils/remove_sample.py -s 0 -r 0.2 -if csv -of csv data/"$dataset".train
  python python-utils/remove_sample.py -s 0 -r 0.5 -if csv -of csv data/"$dataset".train
done


#echo "Downloading higgs from UCI Machine Learning Repository"
#wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz -O data/higgs.gz
#gzip -d data/higgs.gz
#python python-utils/scale_y.py data/higgs -if csv -of csv
#python python-utils/train_test_split.py -v 0 -t 0.2 -s 0 --scale-y data/higgs -if csv -of csv


#echo "Removing samples from datasets"
#for dataset in higgs; do
#  python python-utils/remove_sample.py -s 0 -r 0.01 -if csv -of csv data/"$dataset".train
#  python python-utils/remove_sample.py -s 0 -r 0.001 -if csv -of csv data/"$dataset".train
#done