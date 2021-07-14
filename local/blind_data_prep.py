from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import glob
import os


parser = ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-d', '--destination', type=str)
args = parser.parse_args()


if not os.path.exists(args.destination):
    os.makedirs(args.destination)
    os.system("ln -s "+args.source+" "+args.destination+"/audio")

wav_scp_file = open(args.destination+"/wav.scp",'w')
text_file = open(args.destination+"/text", "w")
cls_file = open(args.destination+"/cls", "w")
utt2spk_file = open(args.destination+"/utt2spk", "w")

for file in tqdm(glob.glob(args.source+"/*.wav")):
	wave_name=file.split('/')[-1]
	wave_path=file
	cls_transcription=".."
	unicode_transcription = ".."
	wav_scp_file.write(wave_name+"  "+wave_path+" \n")
	text_file.write(wave_name+" "+unicode_transcription+"\n")
	cls_file.write(wave_name+" "+cls_transcription+"\n")
	utt2spk_file.write(wave_name+" "+wave_name+"\n")

#os.system("utils/utt2spk_to_spk2utt.pl <"+args.destination+"/utt2spk >"+args.destination+"/spk2utt || exit 1")


utt2spk_file.close()
wav_scp_file.close()
text_file.close()
