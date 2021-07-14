MAIN_ROOT=/common/software/espnet-0.9.6/
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
LD_LIBRARY_PATH=
PYTHONPATH=

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PWD/local/espnet/bin:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
