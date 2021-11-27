import argparse
import torch
from dataset.data import AudioDataLoader, AudioDataset
from src.trainer import Trainer
from model.conv_tasnet import ConvTasNet
from model.dual_path_rnn import Dual_RNN_model
from model.dptnet import DPTNet
from model.sepformer import Sepformer
from model.sudormrf import SuDORMRF
from model.galr import GALR
from model.sandglasset import Sandglasset
from model.snnet_1 import SN_Net
import json5
import numpy as np
from adamp import AdamP, SGDP


def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 数据
    tr_dataset = AudioDataset(json_dir=config["train_dataset"]["train_dir"],  # 目录下包含 mix.json, s1.json, s2.json
                              batch_size=config["train_dataset"]["batch_size"],
                              sample_rate=config["train_dataset"]["sample_rate"],  # 采样率
                              segment=config["train_dataset"]["segment"])  # 语音时长

    cv_dataset = AudioDataset(json_dir=config["validation_dataset"]["validation_dir"],
                              batch_size=config["validation_dataset"]["batch_size"],
                              sample_rate=config["validation_dataset"]["sample_rate"],
                              segment=config["validation_dataset"]["segment"],
                              cv_max_len=config["validation_dataset"]["cv_max_len"])

    tr_loader = AudioDataLoader(tr_dataset,
                                batch_size=config["train_loader"]["batch_size"],
                                shuffle=config["train_loader"]["shuffle"],
                                num_workers=config["train_loader"]["num_workers"])

    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size=config["validation_loader"]["batch_size"],
                                shuffle=config["validation_loader"]["shuffle"],
                                num_workers=config["validation_loader"]["num_workers"])

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # 模型
    if config["model"]["type"] == "conv_tasnet":
        model = ConvTasNet(N=config["model"]["conv_tasnet"]["N"],
                           L=config["model"]["conv_tasnet"]["L"],
                           B=config["model"]["conv_tasnet"]["B"],
                           H=config["model"]["conv_tasnet"]["H"],
                           P=config["model"]["conv_tasnet"]["P"],
                           X=config["model"]["conv_tasnet"]["X"],
                           R=config["model"]["conv_tasnet"]["R"],
                           C=config["model"]["conv_tasnet"]["C"],
                           norm_type=config["model"]["conv_tasnet"]["norm_type"],  # "gLN", "cLN", "BN"
                           causal=config["model"]["conv_tasnet"]["causal"],
                           mask_nonlinear=config["model"]["conv_tasnet"]["mask_nonlinear"])  # "relu", "softmax"
    elif config["model"]["type"] == "dual_path_rnn":
        model = Dual_RNN_model(in_channels=config["model"]["dual_path_rnn"]["in_channels"],
                               out_channels=config["model"]["dual_path_rnn"]["out_channels"],
                               hidden_channels=config["model"]["dual_path_rnn"]["hidden_channels"],
                               kernel_size=config["model"]["dual_path_rnn"]["kernel_size"],
                               rnn_type=config["model"]["dual_path_rnn"]["rnn_type"],
                               norm=config["model"]["dual_path_rnn"]["norm"],
                               dropout=config["model"]["dual_path_rnn"]["dropout"],
                               bidirectional=config["model"]["dual_path_rnn"]["bidirectional"],
                               num_layers=config["model"]["dual_path_rnn"]["num_layers"],
                               K=config["model"]["dual_path_rnn"]["K"],
                               num_spks=config["model"]["dual_path_rnn"]["num_spks"])
    elif config["model"]["type"] == "dptnet":
        model = DPTNet(N=config["model"]["dptnet"]["N"],
                       C=config["model"]["dptnet"]["C"],
                       L=config["model"]["dptnet"]["L"],
                       H=config["model"]["dptnet"]["H"],
                       K=config["model"]["dptnet"]["K"],
                       B=config["model"]["dptnet"]["B"])
    elif config["model"]["type"] == "sepformer":
        model = Sepformer(N=config["model"]["sepformer"]["N"],
                          C=config["model"]["sepformer"]["C"],
                          L=config["model"]["sepformer"]["L"],
                          H=config["model"]["sepformer"]["H"],
                          K=config["model"]["sepformer"]["K"],
                          Global_B=config["model"]["sepformer"]["Global_B"],
                          Local_B=config["model"]["sepformer"]["Local_B"])
    elif config["model"]["type"] == "sudormrf":
        model = SuDORMRF(out_channels=config["model"]["sudormrf"]["out_channels"],
                         in_channels=config["model"]["sudormrf"]["in_channels"],
                         num_blocks=config["model"]["sudormrf"]["num_blocks"],
                         upsampling_depth=config["model"]["sudormrf"]["upsampling_depth"],
                         enc_kernel_size=config["model"]["sudormrf"]["enc_kernel_size"],
                         enc_num_basis=config["model"]["sudormrf"]["enc_num_basis"],
                         num_sources=config["model"]["sudormrf"]["num_sources"])
    elif config["model"]["type"] == "galr":
        model = GALR(in_channels=config["model"]["galr"]["in_channels"],
                     out_channels=config["model"]["galr"]["out_channels"],
                     kernel_size=config["model"]["galr"]["kernel_size"],
                     length=config["model"]["galr"]["length"],
                     hidden_channels=config["model"]["galr"]["hidden_channels"],
                     affine=config["model"]["galr"]["affine"],
                     num_layers=config["model"]["galr"]["num_layers"],
                     bidirectional=config["model"]["galr"]["bidirectional"],
                     num_heads=config["model"]["galr"]["num_heads"],
                     cycle_amount=config["model"]["galr"]["cycle_amount"],
                     speakers=config["model"]["galr"]["speakers"])
    elif config["model"]["type"] == "sandglasset":
        model = Sandglasset(in_channels=config["model"]["sandglasset"]["in_channels"],
                            out_channels=config["model"]["sandglasset"]["out_channels"],
                            kernel_size=config["model"]["galr"]["kernel_size"],
                            length=config["model"]["sandglasset"]["length"],
                            hidden_channels=config["model"]["sandglasset"]["hidden_channels"],
                            num_layers=config["model"]["sandglasset"]["num_layers"],
                            bidirectional=config["model"]["sandglasset"]["bidirectional"],
                            num_heads=config["model"]["sandglasset"]["num_heads"],
                            cycle_amount=config["model"]["sandglasset"]["cycle_amount"],
                            speakers=config["model"]["sandglasset"]["speakers"])
    elif config["model"]["type"] == "snnet":
        model = SN_Net()
    else:
        print("No loaded model!")

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"], config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
    else:
        print("Not support optimizer")
        return

    trainer = Trainer(data, model, optimize, config)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/train/train.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)
