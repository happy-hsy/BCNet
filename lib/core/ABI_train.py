import sys
import os
sys.path.insert(0, os.getcwd())
print(os.getcwd())
import _init_paths
from lib.dataset.dataset_thu_abi import VideoDataSet
from lib.loss.ABI_loss import  abi_loss_func 
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from lib.utils import opts
from lib.model.abi_model import get_abi_model
import time
from lib.utils.log_save import create_logger
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
logger = create_logger('ABI')


def train_ABI(data_loader, model, optimizer, epoch):
    model.train()
    epoch_frame_loss = 0 
    epoch_proposal_loss = 0
    epoch_background_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, sample_mask,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background, index) in enumerate(data_loader):

        batch = sample_mask.shape[0]
        tmp_scale = sample_mask.shape[1]

        input_data = input_data.cuda()
        sample_mask = sample_mask.cuda().reshape(batch,tmp_scale,-1)
        cls_p, cls_b ,confidence_p, confidence_b = model(input_data, sample_mask)
        loss = abi_loss_func(cls_p, cls_b ,confidence_p, confidence_b,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        epoch_frame_loss += loss[1].cpu().detach().numpy()
        epoch_proposal_loss += loss[2].cpu().detach().numpy()
        epoch_background_loss += loss[3].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
    print_info = "ABI training loss(epoch %d): frame_loss: %.03f, action_loss: %.03f, background_loss: %.03f, total_loss: %.03f" % (
            epoch,
            epoch_frame_loss / (n_iter + 1),
            epoch_proposal_loss / (n_iter + 1),
            epoch_background_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))
    logger.info(print_info)


def test_ABI(data_loader, model, epoch):
    model.eval()
    epoch_frame_loss = 0 
    epoch_proposal_loss = 0
    epoch_background_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, sample_mask,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background,index) in enumerate(data_loader):

        batch = sample_mask.shape[0]
        tmp_scale = sample_mask.shape[1]

        input_data = input_data.cuda()
        sample_mask = sample_mask.cuda().reshape(batch,tmp_scale,-1)
        cls_p, cls_b ,confidence_p, confidence_b = model(input_data, sample_mask)
        loss = abi_loss_func(cls_p, cls_b ,confidence_p, confidence_b,match_score_action,match_score_start,match_score_end,match_score_background,gt_proposal,gt_background)
        epoch_frame_loss += loss[1].cpu().detach().numpy()
        epoch_proposal_loss += loss[2].cpu().detach().numpy()
        epoch_background_loss += loss[3].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
    print_info = "ABI testing loss(epoch %d): frame_loss: %.03f, action_loss: %.03f, background_loss: %.03f, total_loss: %.03f" % (
            epoch,
            epoch_frame_loss / (n_iter + 1),
            epoch_proposal_loss / (n_iter + 1),
            epoch_background_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1))
    logger.info(print_info)

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/"+ str(epoch) + "_ABI_checkpoint.pth.tar")

def ABI_Train(opt):
    model = get_abi_model()
    model = torch.nn.DataParallel(model).cuda()


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["abi_batch_size"]
                                               , shuffle=True,
                                               num_workers=8, 
                                               pin_memory=True
                                               )

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="val"),
                                              batch_size=opt["abi_batch_size"]
                                              , shuffle=False,
                                              num_workers=8, 
                                              pin_memory=True
                                            )
    print("data len:",len(train_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    for epoch in range(opt["train_abi_epochs"]):
        scheduler.step()
        train_ABI(train_loader, model, optimizer, epoch)
        test_ABI(test_loader, model, epoch)



def main(opt):
    if opt["mode"] == "train":
        ABI_Train(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/abi_opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    main(opt)
