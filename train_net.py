import os
from lib.config import cfg
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model

def train(cfg, network, strict=False):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume, strict=strict)
    end_epoch = cfg.train.epoch
    save_dir = cfg.model_dir

    train_loader = make_data_loader(cfg, is_train=True)

    for epoch in range(begin_epoch, end_epoch):

        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, save_dir)

    return network

def main():
    if cfg.is_stage:
        for i in range(cfg.stage.num):
            if i == 0:
                cfg.resume = True
                strict = False
            else:
                cfg.resume = True
                strict = False
            cfg.train.epoch = cfg.stage.train_epoch[i]
            cfg.iter = cfg.stage.iter[i]
            cfg.train.batch_size = cfg.stage.batch_size[i]
            network = make_network(cfg)
            train(cfg, network, strict=strict)
    else:
        network = make_network(cfg)
        train(cfg, network)


if __name__ == "__main__":
    main()
