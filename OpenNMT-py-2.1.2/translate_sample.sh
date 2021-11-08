tvt="test"
step="10"
outfile="ex.test/${tvt}_${step}.out"

python translate.py -model ex.test/data/run/model_step_$step.pt \
                    -src ex.test/data/tmp.src \
                    -adj ex.test/data/tmp.adj \
                    -output $outfile \
                    -gpu 1 -verbose

sed -r 's/(@@ )|(@@ ?$)//g' $outfile > $outfile.r

sacrebleu ex.test/data/tmp.trg -i $outfile.r -l en-ja -b