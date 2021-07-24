from argparse import ArgumentParser
import numpy as np
import os


parser = ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-d', '--destination', type=str)

args = parser.parse_args()

if not os.path.exists(args.destination):
    os.makedirs(args.destination)

file = open(args.source+"/files/segments", 'r')

text_file = open(args.destination+"/text", "w")
cls_file = open(args.destination+"/cls", "w")


		

ind=0
for line in file:
	line = line.strip()
	line = line.replace('\t', ' ')
	wave_name=line.split(' ')[0]
	cls_transcription=".."
	c_transcription = ".."
	text_file.write(wave_name+" "+c_transcription+"\n")
	cls_file.write(wave_name+" "+cls_transcription+"\n")
	ind=ind+1


file.close()
#text_file.close()
cls_file.close()


file = open(args.source+"/files/wav.scp", 'r')
wav_scp_file = open(args.destination+"/wav.scp",'w')
for line in file:
	line = line.strip()
	line = line.replace('\t', ' ')
	wave_name=line.split(' ')[0]
	wave_path=args.source+line.split(' ')[1]
	wav_scp_file.write(wave_name+" sox "+wave_path+" -t wav -r 8000 - | \n")
wav_scp_file.close()

os.system("cp "+args.source+"/files/utt2spk "+args.destination+"/utt2spk")
os.system("cp "+args.source+"/files/spk2utt "+args.destination+"/spk2utt")
#os.system("cp "+args.source+"/files/spkr_list "+args.destination+"/spkr_list")
os.system("cp "+args.source+"/files/segments "+args.destination+"/segments")



