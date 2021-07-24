#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



# general configuration
backend=pytorch
stage=1      
stop_stage=2
ngpu=2         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=60
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

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



dumpdir=dump/

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

if [ ! -f "subtask1.zip" ] && [ ! -f "subtask1.zip" ]; then
    echo "Pretrained models not found. \
     Please download the model zip files \
     and place them in below-given folder"
    echo $PWD
    exit -1 
fi

rm -rf exp
mkdir exp
echo "Unpacking pre-trained models"
unzip subtask1.zip -d exp/subtask1
unzip subtask2.zip -d exp/subtask2


train_set=subtask1
expdir=exp/subtask1
dict=${expdir}/${train_set}_units.txt
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
            subtask1_blindtest/audio/  --destination data/blind-test-task1

    fbankdir=${dumpdir}/fbank         
    local/fix_data_dir.sh data/blind-test-task1
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
       data/blind-test-task1 exp/make_fbank/blind-test-task1 ${fbankdir}
    local/fix_data_dir.sh data/blind-test-task1 

    feat_recog_dir=${dumpdir}/blind-test-task1/delta${do_delta}; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/blind-test-task1/feats.scp ${expdir}/cmvn.ark exp/dump_feats/recog/blind-test-task1 \
        ${feat_recog_dir}

    data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type char \
        data/blind-test-task1 ${dict} > ${feat_recog_dir}/data.json 

    
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    #### use CPU for decoding
    ngpu=0

    
    recog_model=model.last${n_average}.avg.best
    opt="--log"


    decode_dir=exp/decode_blind-test-task1_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
    

    lang_model=rnnlm.model.best

    # set batchsize 0 to disable batch decoding
    ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
        asr_dual_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        --result-label ${decode_dir}/data.JOB.json \
        --model ${expdir}/${recog_model}  \
        --model-conf ${expdir}/model.json \
        --api v2 \
        --rnnlm ${expdir}/${lang_model} \
        --rnnlm-conf ${expdir}/rnnlm-model.json

    d_dir=${decode_dir}

    concatjson.py ${d_dir}/data.*.json > ${d_dir}/data.json

    json2trn.py ${d_dir}/data.json ${dict} --num-spkrs 1 --refs ${d_dir}/ref.trn --hyps ${d_dir}/hyp.trn

    sed -i.bak2 -r 's/<blank> //g' ${d_dir}/hyp.trn

    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/ref.trn > ${d_dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/hyp.trn > ${d_dir}/hyp.wrd.trn

    sed 's/-.*//' ${d_dir}/hyp.wrd.trn > ${d_dir}/submission-file.s1
    sed 's/(/\t/' ${d_dir}/submission-file.s1 > ${d_dir}/submission-file.s2
    awk 'BEGIN {FS="\t"; OFS=" "} {print $2, $1}' ${d_dir}/submission-file.s2 > subtask1_submission_file.txt

fi

train_set=subtask2
expdir=exp/subtask2
dict=${expdir}/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${expdir}/${train_set}_${bpemode}${nbpe}
task2_blind_set="Bengali-English Hindi-English"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding Blind Data"
    if [ ! -d "subtask2_blindtest/" ] 
    then
        rm -rf subtask2_blindtest/
        echo "Directory subtask2_blindtest/audio/ DOES NOT exists downloading data."
        wget http://www.ee.iisc.ac.in/people/faculty/prasantg/downloads/subtask2_blindtest.tar.gz
        mkdir -p subtask2_blindtest
        tar -xf subtask2_blindtest.tar.gz -C subtask2_blindtest
    fi

    rm -rf submission-file-task2.txt

    for rtask in ${task2_blind_set}; do
    
        python local/blind_data_prep_task2.py --source \
                /speech/datasets/IndicASRChallenge/blind_test2/${rtask}/  --destination data/blind-test-task2-${rtask}
        fbankdir=${dumpdir}/fbank         
        local/fix_data_dir.sh data/blind-test-task2-${rtask}

        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
        --write_utt2num_frames true data/blind-test-task2-${rtask} \
        exp/make_fbank/blind-test-task2-${rtask} ${fbankdir}
        local/fix_data_dir.sh data/blind-test-task2-${rtask}

        feat_recog_dir=${dumpdir}/blind-test-task2-${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/blind-test-task2-${rtask}/feats.scp \
        ${expdir}/cmvn.ark exp/dump_feats/recog/blind-test-task2-${rtask} \
        ${feat_recog_dir}
 
        data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type char \
        data/blind-test-task2-${rtask} ${dict} > ${feat_recog_dir}/data.json 

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        recog_model=model.val${n_average}.avg.best
        opt="--log ${expdir}/results/log"

        decode_dir=exp/decode_blind-test-task2-${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
            asr_dual_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${decode_dir}/data.JOB.json \
            --model ${expdir}/${recog_model}  \
            --api v2 

        d_dir=${decode_dir}

        concatjson.py ${d_dir}/data.*.json > ${d_dir}/data.json

        json2trn.py ${d_dir}/data.json ${dict} --num-spkrs 1 --refs ${d_dir}/ref.trn --hyps ${d_dir}/hyp.trn

        sed -i.bak2 -r 's/<blank> //g' ${d_dir}/hyp.trn

        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/ref.trn > ${d_dir}/ref.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${d_dir}/hyp.trn > ${d_dir}/hyp.wrd.trn

        sed 's/-.*//' ${d_dir}/hyp.wrd.trn > ${d_dir}/submission-file.s1
        sed 's/(/\t/' ${d_dir}/submission-file.s1 > ${d_dir}/submission-file.s2
        awk 'BEGIN {FS="\t"; OFS=" "} {print $2, $1}' ${d_dir}/submission-file.s2 > submission-file-task2-${rtask}.txt
        cat submission-file-task2-${rtask}.txt >> submission-file-task2.txt
    done            
fi

