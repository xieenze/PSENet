cp ../../../admin/total_text/test.py .
mdl test.py $1
rm test.py
python2 ../../../admin/total_text/Deteval.py
