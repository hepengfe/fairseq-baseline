SCRIPTS=mosesdecoder/scripts
src=en
tgt=de
lang=en-de
OUTDIR=wmt16_en_de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013


CLEAN=$SCRIPTS/training/clean-corpus-n.perl
cd $orig
cd ..

perl $CLEAN  $tmp/bpe.train $src $tgt $prep/train 1 2000
perl $CLEAN  $tmp/bpe.valid $src $tgt $prep/valid 1 2000

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done