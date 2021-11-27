import argparse
import os
import librosa
import torch
from dataset.data import EvalDataLoader, EvalDataset
from model.conv_tasnet import ConvTasNet
from model.dual_path_rnn import Dual_RNN_model
from model.dptnet import DPTNet
from model.galr import GALR
from model.sandglasset import Sandglasset
from src.utils import remove_pad
import json5
import time


def main(config):

    if config["mix_dir"] is None and config["mix_json"] is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, mix_json is ignored.")

    # 加载模型
    if config["model"] == "conv_tasnet":
        model = ConvTasNet.load_model(config["model_path"])
    elif config["model"] == "dual_path_rnn":
        model = Dual_RNN_model.load_model(config["model_path"])
    elif config["model"] == "dptnet":
        model = DPTNet.load_model(config["model_path"])
    elif config["model"] == "galr":
        model = GALR.load_model(config["model_path"])
    elif config["model"] == "sandglasset":
        model = Sandglasset.load_model(config["model_path"])
    else:
        print("No loaded model!")

    model.eval()  # 将模型设置为校验模式

    if torch.cuda.is_available():
        model.cuda()

    # 加载数据
    eval_dataset = EvalDataset(config["mix_dir"],
                               config["mix_json"],
                               batch_size=config["batch_size"],
                               sample_rate=config["sample_rate"])

    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)

    os.makedirs(config["out_dir"], exist_ok=True)
    os.makedirs(config["out_dir"]+"/mix/", exist_ok=True)
    os.makedirs(config["out_dir"]+"/s1/", exist_ok=True)
    os.makedirs(config["out_dir"]+"/s2/", exist_ok=True)

    # 音频生成函数
    def write_wav(inputs, filename, sr=config["sample_rate"]):
        librosa.output.write_wav(filename, inputs, sr)  # norm=True)

    # 不进行反向传播梯度计算
    with torch.no_grad():

        for (i, data) in enumerate(eval_loader):

            print("{}-th Batch Data Start Generate".format(i))

            start_time = time.time()

            mixture, mix_lengths, filenames = data

            if torch.cuda.is_available():

                mixture = mixture.cuda()

                mix_lengths = mix_lengths.cuda()

            estimate_source = model(mixture)  # 将数据放入模型

            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)

            mixture = remove_pad(mixture, mix_lengths)

            for i, filename in enumerate(filenames):

                filename = os.path.join(config["out_dir"]+"/mix/", os.path.basename(filename).strip('.wav'))

                write_wav(mixture[i], filename + '.wav')

                C = flat_estimate[i].shape[0]

                for c in range(C):
                    if c == 0:
                        filename = os.path.join(config["out_dir"]+"/s1/", os.path.basename(filename).strip('.wav'))
                        write_wav(flat_estimate[i][c], filename + '_s{}.wav'.format(c + 1))
                    elif c == 1:
                        filename = os.path.join(config["out_dir"]+"/s2/", os.path.basename(filename).strip('.wav'))
                        write_wav(flat_estimate[i][c], filename + '_s{}.wav'.format(c + 1))
                    else:
                        print("Continue To Add")

            end_time = time.time()

            run_time = end_time - start_time

            print("Elapsed Time: {} s".format(run_time))

        print("Data Generation Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/test/separate.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)
