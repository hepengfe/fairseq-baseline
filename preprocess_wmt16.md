### Download and prepare the data
```
cd examples/translation/
bash prepare-wmt16en2de.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/wmt16_en_de
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt16en2de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60
```
