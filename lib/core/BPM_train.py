import sys
import os
import torch.distributed as dist

sys.path.insert(0, os.getcwd())
import _init_paths
print(os.getcwd())
from lib.dataset.dataset_thu_bpm import VideoDataSet
from lib.loss.BPM_loss import bpm_loss_func
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from lib.utils import opts
from lib.model.bpm_model import get_bpm_model
from lib.utils.log_save import create_logger
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

logger = create_logger('BPM')


def train_bpm(data_loader, model, optimizer, epoch):
    model.train()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_loss = 0
    for n_iter, (input_data,label_action, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()

        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_action = label_action.cuda()

        se = model(input_data)
        action = se[:, :, 0]
        start = se[:, :, 1]
        end = se[:, :, 2]

        loss, action_loss, start_loss, end_loss = bpm_loss_func(action, start, end, label_action, label_start, label_end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_action_loss += action_loss.cpu().detach().numpy()
        epoch_end_loss += end_loss.cpu().detach().numpy()
        epoch_start_loss += start_loss.cpu().detach().numpy()
        epoch_loss += loss.cpu().detach().numpy()
    print_info = "BPM training loss(epoch %d):action_loss: %.03f, start_loss: %.03f, end_loss: %.03f total_loss: %.03f" % (epoch, epoch_action_loss / (n_iter + 1), epoch_start_loss / (n_iter + 1), epoch_end_loss / (n_iter + 1), epoch_loss / (n_iter + 1))

    logger.info(print_info)


def test_bpm(data_loader, model, epoch):
    model.eval()
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_loss = 0
    epoch_action_loss = 0
    for n_iter, (input_data,label_action, label_start, label_end) in enumerate(data_loader):

        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_action = label_action.cuda()

        se = model(input_data)
        action = se[:, :, 0]
        start = se[:, :, 1]
        end = se[:, :, 2]

        loss, action_loss, start_loss, end_loss = bpm_loss_func(action, start, end, label_action, label_start,
                                                                label_end)
        epoch_action_loss += action_loss.cpu().detach().numpy()
        epoch_end_loss += end_loss.cpu().detach().numpy()
        epoch_start_loss += start_loss.cpu().detach().numpy()
        epoch_loss += loss.cpu().detach().numpy()
    print_info = "BPM testing loss(epoch %d):action_loss: %.03f, start_loss: %.03f, end_loss: %.03f total_loss: %.03f" % (epoch, epoch_action_loss / (n_iter + 1), epoch_start_loss / (n_iter + 1), epoch_end_loss / (n_iter + 1),
    epoch_loss / (n_iter + 1))
    logger.info(print_info)

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/"+ str(epoch) + "_BPM_checkpoint.pth.tar")

def BPM_Train(opt):
    model = get_bpm_model()
    model = torch.nn.DataParallel(model).cuda()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt['bpm_batch_size'], shuffle=True,
                                               num_workers=8, pin_memory=True
                                               )

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="val"),
                                              batch_size=opt['bpm_batch_size'],shuffle=False,
                                              num_workers=8, pin_memory=True
                                            )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    for epoch in range(opt["train_bpm_epochs"]):
        scheduler.step()
        train_bpm(train_loader, model, optimizer, epoch)
        test_bpm(test_loader, model, epoch)


def main(opt):
    if opt["mode"] == "train":
        BPM_Train(opt)
    



if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/bpm_opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    main(opt)
