#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



# general configuration
backend=pytorch
stage=1      
stop_stage=1
ngpu=2         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=60
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=exp/train_multi_lingal_pytorch_train_specaug/results/snapshot.ep.18        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=  # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=2                  # the number of ASR models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/speech/datasets/IndicASRChallenge
ai4bharathdir=/speech/datasets/ai4bharat/data
musan_root=/speech/datasets/musan

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
cls_nbpe=800
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=subtask1
train_sp=train_sp
train_dev=dev
recog_set="Tamil-dev Telugu-dev Gujarati-dev Hindi-dev Marathi-dev Odia-dev"

dumpdir=dump/

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

expdir=exp/subtask1
dict=${expdir}/${train_set}_${bpemode}${nbpe}_units.txt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Decoding Blind Data"
    if [ ! -d "subtask1_blindtest/audio/" ] 
    then
        rm -rf subtask1_blindtest/
        echo "Directory subtask1_blindtest/audio/ DOES NOT exists downloading data."
        wget http://www.ee.iisc.ac.in/people/faculty/prasantg/downloads/subtask1_blindtest.tar.gz
        mkdir -p subtask1_blindtest
        tar -xf subtask1_blindtest.tar.gz -C subtask1_blindtest
    fi
    
    python local/blind_data_prep.py --source \
            subtask1_blindtest/audio/  --destination data/blind-test

    fbankdir=${dumpdir}/fbank         
    local/fix_data_dir.sh data/blind-test
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
       data/blind-test exp/make_fbank/blind-test ${fbankdir}
    local/fix_data_dir.sh data/blind-test 

    feat_recog_dir=${dumpdir}/blind-test/delta${do_delta}; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/blind-test/feats.scp ${expdir}/cmvn.ark exp/dump_feats/recog/blind-test \
        ${feat_recog_dir}

    data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type char \
        data/blind-test ${dict} > ${feat_recog_dir}/data.json 

    
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    #### use CPU for decoding
    ngpu=0

    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"
    else
        recog_model=model.last${n_average}.avg.best
        opt="--log"
    fi

    decode_dir=decode_blind-test_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
    

    lang_model=rnnlm.model.best

    # set batchsize 0 to disable batch decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_dual_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/${recog_model}  \
        --model-conf ${expdir}/model.json \
        --api v2 \
        --rnnlm ${expdir}/${lang_model} \
        --rnnlm-conf ${expdir}/rnnlm-model.json

    d_dir=${expdir}/decode_blind-test_${recog_model}_$(basename ${decode_config%.*})_${lmtag}


    concatjson.py ${d_dir}/data.*.json > ${d_dir}/data.json

    json2trn.py ${d_dir}/data.json ${dict} --num-spkrs 1 --refs ${d_dir}/ref.trn --hyps ${d_dir}/hyp.trn

    sed -i.bak2 -r 's/<blank> //g' ${d_dir}/hyp.trn

    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/ref.trn > ${d_dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/hyp.trn > ${d_dir}/hyp.wrd.trn

    sed 's/-.*//' ${d_dir}/hyp.wrd.trn > ${d_dir}/submission-file.s1
    sed 's/(/\t/' ${d_dir}/submission-file.s1 > ${d_dir}/submission-file.s2
    awk 'BEGIN {FS="\t"; OFS=" "} {print $2, $1}' ${d_dir}/submission-file.s2 > ${d_dir}/submission-file

fi

dict=data/lang_1char/${train_set}_units.txt
cls_dict=data/lang_1char/${train_set}_cls_units.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding Blind Data using model 2"
    nj=60
    python local/blind_data_prep.py --source \
            /speech/datasets/IndicASRChallenge/blind_test1/audio/  --destination data/blind-test

    fbankdir=${dumpdir}/fbank         
    local/fix_data_dir.sh data/blind-test
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
       data/blind-test exp/make_fbank/blind-test ${fbankdir}
    local/fix_data_dir.sh data/blind-test 

    feat_recog_dir=${dumpdir}/blind-test/delta${do_delta}; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/blind-test/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/recog/blind-test \
        ${feat_recog_dir}

    data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type char \
        data/blind-test ${dict} > ${feat_recog_dir}/data.json 

    
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    #### use CPU for decoding
    ngpu=0

    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"
    else
        recog_model=model.last${n_average}.avg.best
        opt="--log"
    fi

    decode_dir=decode_blind-test_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
    

    lang_model=rnnlm.model.best

    # set batchsize 0 to disable batch decoding
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_dual_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/${recog_model}  \
        --model-conf ${expdir}/model.json \
        --api v2 \
        --rnnlm ${lmexpdir}/${lang_model} \
        --rnnlm-conf ${expdir}/rnnlm-model.json

    d_dir=${expdir}/decode_blind-test_${recog_model}_$(basename ${decode_config%.*})_${lmtag}


    concatjson.py ${d_dir}/data.*.json > ${d_dir}/data.json

    json2trn.py ${d_dir}/data.json ${dict} --num-spkrs 1 --refs ${d_dir}/ref.trn --hyps ${d_dir}/hyp.trn

    sed -i.bak2 -r 's/<blank> //g' ${d_dir}/hyp.trn

    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/ref.trn > ${d_dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/hyp.trn > ${d_dir}/hyp.wrd.trn

    sed 's/-.*//' ${d_dir}/hyp.wrd.trn > ${d_dir}/submission-file.s1
    sed 's/(/\t/' ${d_dir}/submission-file.s1 > ${d_dir}/submission-file.s2
    awk 'BEGIN {FS="\t"; OFS=" "} {print $2, $1}' ${d_dir}/submission-file.s2 > ${d_dir}/submission-file

fi
