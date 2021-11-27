import argparse
import json
import os
import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):

    file_infos = []
    in_dir = os.path.abspath(in_dir)  # 返回绝对路径
    wav_list = os.listdir(in_dir)  # 返回该目录下文件清单

    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):  # 判断是否以 .wav 结尾
            continue
        wav_path = os.path.join(in_dir, wav_file)  # 拼接路径
        samples, _ = librosa.load(wav_path, sr=sample_rate)  # 读取语音
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):  # 如果输出路径不存在，就创造该路径
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)  # 将信息写入 json


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),  # 拼接路径
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("WSJ0 data preprocessing")

    parser.add_argument('--in-dir',
                        type=str,
                        default="./min",
                        help='Directory path of wsj0 including tr, cv and tt')

    parser.add_argument('--out-dir',
                        type=str,
                        default="./json/",
                        help='Directory path to put output files')

    parser.add_argument('--sample-rate',
                        type=int,
                        default=8000,
                        help='Sample rate of audio file')

    args = parser.parse_args()

    preprocess(args)
