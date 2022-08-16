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

### a sample command

```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 10000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```


### command cache
* max_tokens for one 
```
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py  \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test
```
* max_tokens for one gpu + fp16
```
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_fp16 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --fp16
```
* nce loss + fp16
```
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_nce --label-smoothing 0.1 \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_fp16_nce \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --fp16
```

* label_smoothed_nce + fp32
```
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_nce --label-smoothing 0.1 \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_snce \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test
```

* nce + fp32
```
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test
```

* debugging cmd + nce loss
```
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 500 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test --cpu
```


* adjust lr based on larger bs
    - not using distributed since zero division error
    - fp might be the causes of zero division? No.
    - DDP is cause evaluation issue 
```
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

...

  File "/home/murphy/pengfei_2022/text_gen/fairseq/fairseq/logging/meters.py", line 293, in get_smoothed_value
    return meter.fn(self)
  File "/home/murphy/pengfei_2022/text_gen/fairseq/fairseq/tasks/translation.py", line 443, in compute_bleu
    **smooth,
  File "/home/murphy/miniconda3/envs/text_gen/lib/python3.6/site-packages/sacrebleu/metrics/bleu.py", line 281, in compute_bleu
    return BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len)
  File "/home/murphy/miniconda3/envs/text_gen/lib/python3.6/site-packages/sacrebleu/metrics/bleu.py", line 102, in __init__
    self._verbose += f"ratio = {self.ratio:.3f} hyp_len = {self.sys_len:d} "
  File "/home/murphy/miniconda3/envs/text_gen/lib/python3.6/site-packages/torch/_tensor.py", line 571, in __format__
    return self.item().__format__(format_spec)
ValueError: Unknown format code 'd' for object of type 'float'
```

```
CUDA_VISIBLE_DEVICES=2,3  fairseq-train \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_2_gpu_fp16 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --fp16
```

* two gpus
```
CUDA_VISIBLE_DEVICES=2,3  fairseq-train \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --distributed-world-size 2 --num-workers 8\
    --max-tokens 15000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_2_gpu_fp16 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --fp16
```



## Try to reproduce the CE expr (normal expr)
```
python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_fp16 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test 
```

```

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 27000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 500 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

## Device 0
```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0;
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 500 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

removed     --save-interval-updates 2000 \


```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_slower_lr;
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-6 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_slower_lr \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_slower_lr \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

--lr 5e-6 
device 1



```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout;
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.5 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

higher drop out rate -> 0.5
device 1




```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout;
CUDA_VISIBLE_DEVICES=2 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.8 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_higher_dropout \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

higher drop out rate -> 0.8
device 1


```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_0.1_dropout;
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_0.1_dropout \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_0.1_dropout \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

lower drop out rate -> 0.1
device 1




```
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0 \
    --tensorboard-logdir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```

nvme address  /checkpoints
k_neg = 10




```
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-7 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.2 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_nce_k_neg_30_lr5e-7_weightdecay_0.2 \
    --tensorboard-logdir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_k_neg_30_lr5e-7_weightdecay_0.2 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```
even higher weight decay (0.2)

```
CUDA_VISIBLE_DEVICES= python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-7 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.2 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_nce_k_neg_30_lr5e-7_weightdecay_0.2 \
    --tensorboard-logdir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_k_neg_30_lr5e-7_weightdecay_0.2 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --cpu
```

even higher weight decay (0.2)
test on cpu

```
CUDA_VISIBLE_DEVICES= python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_ce_loss_debug \
    --tensorboard-logdir home/murphy/checkpoints/transformer_wmt16_en_de_1_gpu_ce_loss_debug \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test --cpu
```
check the speed of ce loss



## Device 1
```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device1;
CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device1 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device1 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

## Device 2
```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_device2;
CUDA_VISIBLE_DEVICES=2 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 18000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 2000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device2 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_device2 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
``` 

### device
```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_15
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 10000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_15 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_15 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```


```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_5
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 10000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_5 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```




```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_200
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion nce \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 10000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_200 \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_nce_debug_k_200 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```

```
rm -rf checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train.py \
    data-bin/wmt16en2de \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-tokens 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-interval-updates 10000 \
    --save-dir checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug \
    --tensorboard-logdir checkpoints/transformer_wmt16_en_de_1_gpu_ce_debug \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```


# Evaluate
fairseq-generate \
    data-bin/wmt16en2de \
    --path checkpoints/transformer_wmt16_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

# data processing results 
```
2022-01-18 19:40:04 | INFO | fairseq_cli.preprocess | [de] Dictionary: 34856 types
2022-01-18 19:40:42 | INFO | fairseq_cli.preprocess | [de] examples/translation/wmt16_en_de/train.de: 3704623 sents, 114107838 tokens, 0.0% replaced (by <unk>)
2022-01-18 19:40:42 | INFO | fairseq_cli.preprocess | [de] Dictionary: 34856 types
2022-01-18 19:40:43 | INFO | fairseq_cli.preprocess | [de] examples/translation/wmt16_en_de/valid.de: 37449 sents, 1154995 tokens, 0.00104% replaced (by <unk>)
2022-01-18 19:40:43 | INFO | fairseq_cli.preprocess | [de] Dictionary: 34856 types
2022-01-18 19:40:44 | INFO | fairseq_cli.preprocess | [de] examples/translation/wmt16_en_de/test.de: 3003 sents, 87408 tokens, 0.879% replaced (by <unk>)
2022-01-18 19:40:44 | INFO | fairseq_cli.preprocess | [en] Dictionary: 33544 types
2022-01-18 19:41:19 | INFO | fairseq_cli.preprocess | [en] examples/translation/wmt16_en_de/train.en: 3704623 sents, 110977942 tokens, 0.0% replaced (by <unk>)
2022-01-18 19:41:19 | INFO | fairseq_cli.preprocess | [en] Dictionary: 33544 types
2022-01-18 19:41:21 | INFO | fairseq_cli.preprocess | [en] examples/translation/wmt16_en_de/valid.en: 37449 sents, 1122037 tokens, 0.00232% replaced (by <unk>)
2022-01-18 19:41:21 | INFO | fairseq_cli.preprocess | [en] Dictionary: 33544 types
2022-01-18 19:41:22 | INFO | fairseq_cli.preprocess | [en] examples/translation/wmt16_en_de/test.en: 3003 sents, 83050 tokens, 0.0012% replaced (by <unk>)
2022-01-18 19:41:22 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to data-bin/wmt16en2de
```


### commonly used link
[architecture](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/models/transformer/transformer_legacy.py#L169)
[fairseq nmt](https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md)
[fairseq-train hyperparameter](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train)





CUDA_VISIBLE_DEVICES=2 fairseq-generate \
    data-bin/wmt16en2de \
    --path checkpoints/transformer_wmt16_en_de_1_gpu_nce_device2/checkpoint_best.pt \
    --beam 5 --remove-bpe


CUDA_VISIBLE_DEVICES= fairseq-generate \
    data-bin/wmt16en2de \
    --path checkpoints/transformer_wmt16_en_de_1_gpu_nce_device0_slower_lr/checkpoint_best.pt \
    --beam 5 --remove-bpe

