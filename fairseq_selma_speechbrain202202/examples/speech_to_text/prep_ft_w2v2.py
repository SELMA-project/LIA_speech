"""
This script prepares data for supervised/self-supervised fine-tuning with wav2vec 2.0
- Input: .tsv files which are outputs of the prep_${dataset}_data.py script
- Output: .tsv, .ltr, .wrd, and .txt (to learn dictitionary) files which are 
inputs for fine-tuning with wav2vec 2.0     
"""

import argparse
import os
import soundfile as sf
from examples.speech_to_text.data_utils import load_df_from_tsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=str, 
                                help="Path to directory where audio files are saved.")
    parser.add_argument("--tsv-path", type=str, 
                                help="Path to input tsv file, which is the output of prep_dataset_data.py")
    parser.add_argument("--dest", type=str, 
                                help="Path to directory where outputs are saved.")
    parser.add_argument("--do-lower", action="store_true",
                                help="Do lower-case if true. Otherwise, do upper-case.")

    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    translations = {}

    split = os.path.basename(args.tsv_path).replace(".tsv", "").split("_")[0]
    df = load_df_from_tsv(args.tsv_path)

    ids_arr = [f"{'_'.join(n.split('_')[:-1])}_{str(n.split('_')[-1]).zfill(4)}.wav" for n in df["id"].values]
    nframes_arr = []
    for idx in ids_arr:
        fname = os.path.join(os.path.join(args.audio_root), idx).replace("_0000.wav", ".wav")
        #print(fname)
        nframes_arr.append(sf.info(fname).frames)

    # tsv file
    with open(os.path.join(args.dest, f"{split}.tsv"), "w") as f:
        print(args.audio_root, file=f)
        for i, id in enumerate(ids_arr):
            id = id.replace("_0000.wav", ".wav")
            print(
                f"{id}\t{nframes_arr[i]}", file=f
            )

    # ltr and wrd file
    with open(args.tsv_path, "r") as f_in, \
        open(os.path.join(args.dest, f"{split}.ltr"), "w") as ltr_out, \
        open(os.path.join(args.dest, f"{split}.wrd"), "w") as wrd_out:
        header = next(f_in).strip()
        for line in f_in:
            line = line.strip().split('\t')
            id = line[0]
            if id not in translations:
                if args.do_lower:
                    texts = line[3].lower() # src_text
                else:
                    texts = line[3] #.upper()
                translations[id] = texts
            print(translations[id], file=wrd_out)
            print(
                " ".join(list(translations[id].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()