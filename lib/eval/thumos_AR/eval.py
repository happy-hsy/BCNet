# -*- coding: utf-8 -*-
import sys
# sys.path.append('./')
#from eval_proposal import ANETproposal
import matplotlib.pyplot as plt
import numpy as np
import csv
from lib.eval.thumos_AR.eval_proposal_thu import ANETproposal



def caculate_proposal(opt):
    print("is eval proposal fun")
    ground_truth_filename = opt['video_info'] + "/thumos_anno_eval.json"
    anet_proposal = ANETproposal(ground_truth_filename, opt['result_file'],
                                 tiou_thresholds=np.linspace(0.5, 1.0, 11),
                                 max_avg_nr_proposals=1000,
                                 subset='test', verbose=True, check_status=False)
    anet_proposal.evaluate()
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    # names = ['AR@50', 'AR@100', 'AR@200', 'AR@500', 'AR@1000']
    # values = [np.mean(recall[:,i]) for i in [49, 99, 199, 499, 999]]
    return (average_nr_proposals, average_recall, recall)

def plot_metric(opt,average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 1.0, 11)):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1,1,1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #plt.show()    
    plt.savefig(opt["save_fig_path"])

def evaluation_proposal(opt):


    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = caculate_proposal(opt)   
    # plot_metric(opt,uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    print ("AR@50 is \t",np.mean(uniform_recall_valid[:,49]))
    print ("AR@100 is \t",np.mean(uniform_recall_valid[:,99]))
    print ("AR@200 is \t",np.mean(uniform_recall_valid[:,199]))
    print ("AR@500 is \t",np.mean(uniform_recall_valid[:,499]))
    print ("AR@1000 is \t",np.mean(uniform_recall_valid[:,999]))
    
