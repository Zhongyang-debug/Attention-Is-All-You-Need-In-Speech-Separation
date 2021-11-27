import argparse
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from dataset.data import AudioDataLoader, AudioDataset
from src.pit_criterion import cal_loss
from model.sepformer import Sepformer
from src.utils import remove_pad
import json5


def cal_SDRi(src_ref, src_est, mix):

    """
        Calculate Source-to-Distortion Ratio improvement (SDRi).

        NOTE: bss_eval_sources is very very slow.

        Args:
            src_ref: numpy.ndarray, [C, T]
            src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
            mix: numpy.ndarray, [T]

        Returns:
            average_SDRi
    """

    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0] - sdr0[0]) + (sdr[1] - sdr0[1])) / 2

    return avg_SDRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):

    """
        Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)

        Args:
            ref_sig: numpy.ndarray, [T]
            out_sig: numpy.ndarray, [T]
        Returns:
            SISNR
    """

    assert len(ref_sig) == len(out_sig)

    ref_sig = ref_sig - np.mean(ref_sig)

    out_sig = out_sig - np.mean(out_sig)

    ref_energy = np.sum(ref_sig ** 2) + eps

    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy

    noise = out_sig - proj

    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)

    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)

    return sisnr


def cal_SISNRi(src_ref, src_est, mix):

    """
        Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)

        Args:
            src_ref: numpy.ndarray, [C, T]
            src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
            mix: numpy.ndarray, [T]
        Returns:
            average_SISNRi
    """

    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])

    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)

    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2

    return avg_SISNRi


def main(config):

    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # 加载模型
    if config["model"] == "sepformer":
        model = Sepformer.load_model(config["model_path"])
    else:
        print("No loaded model!")

    model.eval()  # 将模型设置为验证模式

    if torch.cuda.is_available():
        model.cuda()

    # 加载数据
    dataset = AudioDataset(config["evaluate_dataset"]["data_dir"],
                           config["evaluate_dataset"]["batch_size"],
                           sample_rate=config["evaluate_dataset"]["sample_rate"],
                           segment=config["evaluate_dataset"]["segment"])

    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    # 不计算梯度
    with torch.no_grad():

        for i, (data) in enumerate(data_loader):

            # torch.Size([1, 32000]) torch.Size([1]) torch.Size([1, 2, 32000])
            padded_mixture, mixture_lengths, padded_source = data

            # 利用 GPU 运算
            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            # torch.Size([1, 2, 32000]) => torch.Size([1, 2, 32000])
            estimate_source = model(padded_mixture)  # 将数据放入模型

            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(padded_source,    # mix
                                                                               estimate_source,  # [s1, s2]
                                                                               mixture_lengths)  # length

            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)

            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1, ": ")

                # Compute SDRi
                if config["cal_sdr"]:
                    # (2, 32000) (2, 32000) (32000,)
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    print("    SDRi = {0:.2f}, ".format(avg_SDRi))
                    total_SDRi += avg_SDRi

                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("    SI-SNRi = {0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi

                total_cnt += 1

    if config["cal_sdr"]:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi/total_cnt))

    print("Average SI_SNR improvement: {0:.2f}".format(total_SISNRi/total_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Speech Separation Performance")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/test/evaluate.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)
