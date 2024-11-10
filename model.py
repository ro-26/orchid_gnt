import torch
import os
from orchid_gnt.transformer_network import GNT
from orchid_gnt.feature_network import ResUNet

def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class GNTModel(object):
    def __init__(self,load_opt=True, load_scheduler=True):
        device = torch.device("cuda:{}".format(0))
        self.coarse_feat_dim=32 
        self.fine_feat_dim=32
        self.N_importance=0
        self.single_net=True
        self.lrate_feature=0.001
        self.lrate_gnt=0.0005
        self.lrate_decay_steps=50000
        self.local_rank=0
        self.lrate_decay_factor=0.5
        self.no_reload=False
        self.ckpt_path=""
        self.distributed=False
        # create coarse GNT
        self.net_coarse = GNT(
            in_feat_ch=self.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=self.N_importance > 0,
        ).to(device)
        # single_net - trains single network which can be used for both coarse and fine sampling
        if self.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                in_feat_ch=self.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.coarse_feat_dim,
            fine_out_ch=self.fine_feat_dim,
            single_net=self.single_net,
        ).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": self.lrate_feature},
                ],
                lr=self.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": self.lrate_feature},
                ],
                lr=self.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.lrate_decay_steps, gamma=self.lrate_decay_factor
        )

        out_folder = "./out/gnt_orchid"
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if self.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[self.local_rank], output_device=self.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[self.local_rank], output_device=self.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[self.local_rank], output_device=self.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }

        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.local_rank))
        else:
            to_load = torch.load(filename)
        # print(to_load["net_coarse"].keys())
        # exit()
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"])

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.ckpt_path):  # load the specified ckpt
                ckpts = [self.ckpt_path]

        if len(ckpts) > 0 and not self.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step
