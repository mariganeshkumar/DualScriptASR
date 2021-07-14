#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



# general configuration
backend=pytorch
stage=11      
stop_stage=12
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

train_set=train_multi_lingual
train_sp=train_sp
train_dev=dev
recog_set="Tamil-dev Telugu-dev Gujarati-dev Hindi-dev Marathi-dev Odia-dev"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage 1: Dictionary preparation"

    mkdir -p dictionaries

    lang_code=( "ta" "te" "gu" )
    lang_name=( "Tamil" "Telugu" "Gujarati" )
    for i in "${!lang_code[@]}"; do

        python local/get_unique_words.py -o dictionaries/${lang_name[i]}_unique_word.txt \
        -f ${datadir}/microsoftspeechcorpusindianlanguages/${lang_code[i]}-in-Train/transcription.txt \
        ${datadir}/microsoftspeechcorpusindianlanguages/${lang_code[i]}-in-Test/transcription.txt
        
        local/phonify_text.sh dictionaries/${lang_name[i]}_unique_word.txt \
        dictionaries/${lang_name[i]}.dict ${nj} 2> dictionaries/${lang_name[i]}.log

        local/clean_csl_dictionary.sh dictionaries/${lang_name[i]}.dict
    done


    lang_code=( "or" "hi" "mr"  )
    lang_name=( "Odia" "Hindi" "Marathi"  )
    for i in "${!lang_code[@]}"; do

        python local/get_unique_words.py -o dictionaries/${lang_name[i]}_unique_word.txt \
        -f ${datadir}/${lang_name[i]}/train/transcription.txt ${datadir}/${lang_name[i]}/test/transcription.txt
        
        local/phonify_text.sh dictionaries/${lang_name[i]}_unique_word.txt \
        dictionaries/${lang_name[i]}.dict ${nj} 2> dictionaries/${lang_name[i]}.log
        
        local/clean_csl_dictionary.sh dictionaries/${lang_name[i]}.dict
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    mkdir -p dictionaries

    lang_code=( "ta" "te" "gu" )
    lang_name=( "Tamil" "Telugu" "Gujarati" )
    for i in "${!lang_code[@]}"; do

        python local/msr_data_prep.py --source \
            ${datadir}/microsoftspeechcorpusindianlanguages/${lang_code[i]}-in-Train --prefix ${lang_name[i]} \
            --cls-mapping dictionaries/${lang_name[i]}.dict --destination data/${lang_name[i]}-train
        python local/msr_data_prep.py --source \
            ${datadir}/microsoftspeechcorpusindianlanguages/${lang_code[i]}-in-Test  --prefix ${lang_name[i]} \
            --cls-mapping dictionaries/${lang_name[i]}.dict --destination data/${lang_name[i]}-dev
    done

    lang_code=( "or" "hi" "mr"  )
    lang_name=( "Odia" "Hindi" "Marathi"  )
    for i in "${!lang_code[@]}"; do

        # use underscore-separated names in data directories.
        python local/indic_data_prep.py --source ${datadir}/${lang_name[i]}/train --prefix ${lang_name[i]}\
            --cls-mapping dictionaries/${lang_name[i]}.dict --destination data/${lang_name[i]}-train 
        python local/indic_data_prep.py --source ${datadir}/${lang_name[i]}/test --prefix ${lang_name[i]}\
            --cls-mapping dictionaries/${lang_name[i]}.dict --destination data/${lang_name[i]}-dev    
    done
    
fi





. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_sp_dir=${dumpdir}/${train_sp}/delta${do_delta}; mkdir -p ${feat_sp_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    fbankdir=${dumpdir}/fbank
    #Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in Tamil-train Tamil-dev Telugu-train Telugu-dev Gujarati-train Gujarati-dev Hindi-train Hindi-dev Marathi-train Marathi-dev Odia-train Odia-dev; do
        local/fix_data_dir.sh data/${x}
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        local/fix_data_dir.sh data/${x}
    done

    local/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/Tamil-train data/Telugu-train data/Gujarati-train data/Hindi-train data/Marathi-train data/Odia-train
    local/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/Tamil-dev data/Telugu-dev data/Gujarati-dev data/Hindi-dev data/Marathi-dev data/Odia-dev
    #utils/data/get_utt2dur.sh data/${train_set}_org
    local/perturb_data_dir_speed.sh 0.9  data/${train_set}_org  data/temp1
    local/perturb_data_dir_speed.sh 1.0  data/${train_set}_org  data/temp2
    local/perturb_data_dir_speed.sh 1.1  data/${train_set}_org  data/temp3

    # frame_shift=0.01
    # awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${train_set}_org/utt2num_frames > data/${train_set}_org/reco2dur

    # if [ ! -d "RIRS_NOISES" ]; then
    # # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    # wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    # unzip rirs_noises.zip
    # fi

    # # Make a version with reverberated speech
    # rvb_opts=()
    # rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    # rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    # # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
    # # additive noise here.
    # steps/data/reverberate_data_dir.py \
    # "${rvb_opts[@]}" \
    # --speech-rvb-probability 1 \
    # --pointsource-noise-addition-probability 0 \
    # --isotropic-noise-addition-probability 0 \
    # --num-replications 1 \
    # --source-sampling-rate 8000 \
    # data/${train_set}_org data/train_reverb
    # cp data/${train_set}_org/cls data/train_reverb/cls
    # cp data/${train_set}_org/utt2dur data/train_reverb/utt2dur
    # local/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
    # rm -rf data/train_reverb
    # mv data/train_reverb.new data/train_reverb

    # # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # # suitable for augmentation.
    # steps/data/make_musan.sh --sampling-rate 8000 $musan_root data

    # # Get the duration of the MUSAN recordings.  This will be used by the
    # # script augment_data_dir.py.
    # for name in speech noise music; do
    # utils/data/get_utt2dur.sh data/musan_${name}
    # mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    # done

    # # Augment with musan_noise
    # local/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${train_set}_org data/train_noise
    # # Augment with musan_music
    # local/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${train_set}_org data/train_music
    # # Augment with musan_speech
    # local/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${train_set}_org data/train_babble

    # # Combine reverb, noise, music, and babble into one directory.
    local/combine_data.sh --extra-files utt2uniq data/${train_sp}_org data/temp1 data/temp2 data/temp3
    #local/subset_data_dir.sh data/train_aug 2000000 data/${train_sp}_org
   
    local/fix_data_dir.sh data/${train_sp}_org
    
    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    local/reduce_data_dir.sh data/${train_set}_org data/${train_set}/tmp/reclist data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_sp}_org data/${train_sp}
    local/reduce_data_dir.sh data/${train_sp}_org data/${train_sp}/tmp/reclist data/${train_sp}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}
    local/reduce_data_dir.sh data/${train_dev}_org data/${train_dev}/tmp/reclist data/${train_dev}

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj  --write_utt2num_frames true \
            data/train_sp  exp/make_fbank/train_sp  ${fbankdir}
    local/fix_data_dir.sh data/train_sp
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_sp}/feats.scp data/${train_sp}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_sp}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/train ${feat_sp_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi



dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
cls_dict=data/lang_char/${train_set}_${bpemode}${cls_nbpe}_cls_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
cls_bpemodel=data/lang_char/${train_set}_${bpemode}${cls_nbpe}_cls
echo "dictionary: ${dict}"
echo "common label set dictionary: ${cls_dict}"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<unk> 1" > ${cls_dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    
    cut -f 2- -d" " data/${train_set}/text | sort | uniq -d > data/lang_char/input.txt
    cut -f 2- -d" " data/${train_set}/cls | sort | uniq -d > data/lang_char/cls_input.txt

    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 #--hard_vocab_limit=false
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    spm_train --input=data/lang_char/cls_input.txt --vocab_size=${cls_nbpe} --model_type=${bpemode} --model_prefix=${cls_bpemodel} --input_sentence_size=100000000 #--hard_vocab_limit=false
    spm_encode --model=${cls_bpemodel}.model --output_format=piece < data/lang_char/cls_input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${cls_dict}

    # make json labels
    local/espnet/utils/data2json.sh --nj ${nj} --feat ${feat_sp_dir}/feats.scp --bpecode ${bpemodel}.model \
        --clsbpecode ${cls_bpemodel}.model data/${train_sp} ${dict} ${cls_dict} > ${feat_sp_dir}/data_${bpemode}${nbpe}.json
    local/espnet/utils/data2json.sh --nj ${nj} --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        --clsbpecode ${cls_bpemodel}.model data/${train_dev} ${dict} ${cls_dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj ${nj} --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi


# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 3: LM Preparation"
    echo ${lmexpdir}
    lmdatadir=dump/lm_local/lm_train_${bpemode}${nbpe}

    
    # lmdatadir=data/local/lm_train

    # if [  -f data/lang_char/text ]; then
    #         rm data/lang_char/all_text
    # fi

    lang_code=( "ta" "te"  "gu"  "hi" "mr" "or")
    lang_name=( "Tamil" "Telugu" "Gujarati"  "Hindi" "Marathi" "Odia"  )
    for i in "${!lang_code[@]}"; do
        cat data/${lang_name[i]}-train/text | sort | uniq -d >> data/lang_char/all_text
        cat /speech/datasets/text/unicode_cls_parallel_text/${lang_name[i]}.txt >> data/lang_char/all_text
    done 

    

    wc -l data/lang_char/text
    

    if [ ! -e ${lmdatadir} ]; then
        mkdir -p ${lmdatadir}
        #shuf data/lang_char/all_text data/lang_char/text
        cut -f 2- -d" " data/lang_char/text |\
            spm_encode --model=${bpemodel}.model --output_format=piece |\
             awk -v maxchars="600" '{ if (NF - 1 < maxchars) print }' | awk '{print $0}' > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
                                                            > ${lmdatadir}/valid.txt
    fi

    #wc -l ${lmdatadir}/train.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
echo $PYTHONPATH
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_dual_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_sp_dir}/data_${bpemode}${nbpe}.json\
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json 
fi

nj=10;

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_dual_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --api v2 \
            --rnnlm ${lmexpdir}/${lang_model} \
            

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding Blind Data"
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
        --model ${expdir}/results/${recog_model}  \
        --api v2 \
        --rnnlm ${lmexpdir}/${lang_model} 

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
echo "dictionary: ${dict}"
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 7: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<unk> 1" > ${cls_dict}
    
    text2token.py -s 1 -n 1 data/${train_sp}/text --trans_type char | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    text2token.py -s 1 -n 1 data/${train_sp}/cls --trans_type char | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${cls_dict}
    wc -l ${cls_dict}

    # make json labels
    local/espnet/utils/data2json.sh --feat ${feat_sp_dir}/feats.scp --trans_type char \
    data/${train_sp} ${dict} ${cls_dict} > ${feat_sp_dir}/data.json
    local/espnet/utils/data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type char \
    data/${train_dev} ${dict} ${cls_dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type char \
        data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi





tag="CharLevel"
resume=/cbr/mari/projects/IS21_ASR/v2/DualLossASR/exp/train_multi_lingual_pytorch_CharLevel/results/snapshot.ep.20

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
echo $PYTHONPATH
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 8: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_dual_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_sp_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --maxlen-out 300

        # 
fi


lmtag="CharLevel"
# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    # echo "stage 9: LM Preparation"

    lmdatadir=data/local/lm_train

    if [  -f data/lang_char/text ]; then
            rm data/lang_char/all_text
    fi

    lang_code=( "ta" "te"  "gu"  "hi" "mr" "or")
    lang_name=( "Tamil" "Telugu" "Gujarati"  "Hindi" "Marathi" "Odia"  )
    for i in "${!lang_code[@]}"; do
        cat data/${lang_name[i]}-train/text | sort | uniq -d >> data/lang_char/all_text
        cat /speech/datasets/text/unicode_cls_parallel_text/${lang_name[i]}.txt >> data/lang_char/all_text
    done 

    

    wc -l data/lang_char/all_text
    
    
    #rm -r $lmdatadir

    if [ ! -e ${lmdatadir} ]; then
        mkdir -p ${lmdatadir}
        shuf  data/lang_char/all_text > data/lang_char/text
        text2token.py -s 0 -n 1 <(cut -f 2- -d" " data/lang_char/text) --trans_type char | \
        awk -v maxchars="600" '{ if (NF - 1 < maxchars) print }' | awk '{print $0}' > ${lmdatadir}/train.txt

        text2token.py -s 0 -n 1 <(cut -f 2- -d" " data/${train_dev}/text) --trans_type char > \
        ${lmdatadir}/valid.txt
    fi

    wc -l ${lmdatadir}/train.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi



if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 10: Decoding"
    nj=10
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_dual_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --api v2 \
            --rnnlm ${lmexpdir}/${lang_model} \
            

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 11: Decoding Blind Data"
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
        --model ${expdir}/results/${recog_model}  \
        --api v2 \
        --rnnlm ${lmexpdir}/${lang_model} 

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
