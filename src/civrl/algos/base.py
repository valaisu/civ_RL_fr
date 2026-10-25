import torch


class Algo:
    lr = 3e-4  # learning rate when --lr is not given
    minibatch = 512  # when --minibatch is not given
    progress = 0.0  # share of training done (by --steps or --minutes, whichever is further); set by train.py

    def __init__(self, args, net, runner):
        self.args = args
        self.net = net
        self.runner = runner
        self.opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    def update(self):
        """Collect experience and improve the network once: returns (decisions, stats)."""
        raise NotImplementedError

    def describe(self):
        """One line of settings for the start-of-run print."""
        raise NotImplementedError

    def on_phase_change(self):
        """Called when --alternate switches the learning side."""
