###################
# Finetune on NLI #
###################
# Finetune English LM on English NLI
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.000001 -n "en" -o "ckpts/roberta-base.xnli-en.1" --repo-id "vocabtrimmer/roberta-base.xnli-en.1"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.000005 -n "en" -o "ckpts/roberta-base.xnli-en.2" --repo-id "vocabtrimmer/roberta-base.xnli-en.2"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.00001 -n "en" -o "ckpts/roberta-base.xnli-en.3" --repo-id "vocabtrimmer/roberta-base.xnli-en.3"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.00005 -n "en" -o "ckpts/roberta-base.xnli-en.4" --repo-id "vocabtrimmer/roberta-base.xnli-en.4"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.0001 -n "en" -o "ckpts/roberta-base.xnli-en.5" --repo-id "vocabtrimmer/roberta-base.xnli-en.5"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.0005 -n "en" -o "ckpts/roberta-base.xnli-en.6" --repo-id "vocabtrimmer/roberta-base.xnli-en.6"
python experiments_zero/finetune_xnli.py -m "roberta-base" --lr 0.001 -n "en" -o "ckpts/roberta-base.xnli-en.7" --repo-id "vocabtrimmer/roberta-base.xnli-en.7"
# Finetune Chinese LM on Chinese NLI
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.000001 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.1" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.1"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.000005 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.2" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.2"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.00001 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.3" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.3"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.00005 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.4" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.4"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.0001 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.5" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.5"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.0005 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.6" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.6"
python experiments_zero/finetune_xnli.py -m "hfl/chinese-roberta-wwm-ext" --lr 0.001 -n "zh" -o "ckpts/chinese-roberta-wwm-ext.xnli-zh.7" --repo-id "vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.7"
# Finetune French LM on French NLI
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.000001 -n "fr" -o "ckpts/camembert-base.xnli-fr.1" --repo-id "vocabtrimmer/camembert-base.xnli-fr.1"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.000005 -n "fr" -o "ckpts/camembert-base.xnli-fr.2" --repo-id "vocabtrimmer/camembert-base.xnli-fr.2"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.00001 -n "fr" -o "ckpts/camembert-base.xnli-fr.3" --repo-id "vocabtrimmer/camembert-base.xnli-fr.3"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.00005 -n "fr" -o "ckpts/camembert-base.xnli-fr.4" --repo-id "vocabtrimmer/camembert-base.xnli-fr.4"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.0001 -n "fr" -o "ckpts/camembert-base.xnli-fr.5" --repo-id "vocabtrimmer/camembert-base.xnli-fr.5"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.0005 -n "fr" -o "ckpts/camembert-base.xnli-fr.6" --repo-id "vocabtrimmer/camembert-base.xnli-fr.6"
python experiments_zero/finetune_xnli.py -m "camembert-base" --lr 0.001 -n "fr" -o "ckpts/camembert-base.xnli-fr.7" --repo-id "vocabtrimmer/camembert-base.xnli-fr.7"
# Finetune Spanish LM on Spanish NLI
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.000001 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.1" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.1"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.000005 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.2" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.2"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.00001 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.3" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.3"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.00005 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.4" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.4"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.0001 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.5" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.5"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.0005 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.6" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.6"
python experiments_zero/finetune_xnli.py -m "dccuchile/bert-base-spanish-wwm-cased" --lr 0.001 -n "es" -o "ckpts/bert-base-spanish-wwm-cased.xnli-es.7" --repo-id "vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.7"

#########################
# Train Vocab Embedding #
#########################
# Train Chinese Vocab from English
MODEL="vocabtrimmer/camembert-base.xnli-fr.5"
LA="fr"
MODEL="vocabtrimmer/chinese-roberta-wwm-ext.xnli-zh.5"
LA="zh"
MODEL="vocabtrimmer/bert-base-spanish-wwm-cased.xnli-es.5"
LA="es"
EN_MODEL="vocabtrimmer/roberta-base.xnli-en.5"
python experiments_zero/train_tokenizer.py -r "${EN_MODEL}" --dataset-config "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained"
python experiments_zero/train_vocab_embedding.py -b 256 -e 3 --lr 0.0001 -c "en-${LA}" -s "${EN_MODEL}" -t "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained" -o "ckpts/roberta-base.xnli-en.swap_${LA}.1" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.1"
python experiments_zero/finetune_xnli.py -m "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.1" --skip-train -n "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}.1" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.1"
python experiments_zero/train_vocab_embedding.py -b 256 -e 3 --lr 0.0005 -c "en-${LA}" -s "${EN_MODEL}" -t "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained" -o "ckpts/roberta-base.xnli-en.swap_${LA}.2" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.2"
python experiments_zero/finetune_xnli.py -m "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.2" --skip-train -n "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}.2" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.2"
python experiments_zero/train_vocab_embedding.py -b 256 -e 3 --lr 0.001 -c "en-${LA}" -s "${EN_MODEL}" -t "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained" -o "ckpts/roberta-base.xnli-en.swap_${LA}.3" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.3"
python experiments_zero/finetune_xnli.py -m "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.3" --skip-train -n "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}.3" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.3"
python experiments_zero/train_vocab_embedding.py -b 256 -e 3 --lr 0.005 -c "en-${LA}" -s "${EN_MODEL}" -t "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained" -o "ckpts/roberta-base.xnli-en.swap_${LA}.4" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.4"
python experiments_zero/finetune_xnli.py -m "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.4" --skip-train -n "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}.4" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.4"
python experiments_zero/train_vocab_embedding.py -b 256 -e 3 --lr 0.01 -c "en-${LA}" -s "${EN_MODEL}" -t "vocabtrimmer/roberta-base.xnli-en.swap_${LA}_tokenizer.untrained" -o "ckpts/roberta-base.xnli-en.swap_${LA}.5" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.5"
python experiments_zero/finetune_xnli.py -m "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.5" --skip-train -n "${LA}" -o "ckpts/roberta-base.xnli-en.swap_${LA}.5" --repo-id "vocabtrimmer/roberta-base.xnli-en.swap_${LA}.5"
