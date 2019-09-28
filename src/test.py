import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import ActivDataset, loader
from models.focal_loss import FocalLoss
from models.ete_waveform import EteWave
from models.post_process import as_seaquence


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def main(args):
    model = EteWave(args.n_class).to(device)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    test_data_file_names =\
        [line.rstrip() for line in open(args.test_data_file_pointer_path)]

    test_dataset = ActivDataset(test_data_file_names, args.root_dir,
                                seq_len=args.test_seq_len, time_step=args.time_step,
                                is_train=False)
    test_loader = loader(test_dataset, 1, shuffle=False)
    test(args, model, test_loader)


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    segmentation_correct = 0
    lack_classifier_correct = 0
    total_len = 0
    lack_total_len = 0
    
    true_seq_labels = []
    inf_seq_labels = []

    true_finish_labels = []
    inf_finish_labels = []
    inf_finish_proba = []

    true_finish_labels_mat = np.empty([len(data_loader), 5])
    inf_finish_labels_mat = np.empty([len(data_loader), 5])
    
    with torch.no_grad():
        for i_batch, (l_data, l_target, l_lack_labels) in enumerate(data_loader):
            l_data = l_data.to(device)
            l_target = l_target.to(device)
            l_lack_labels = l_lack_labels.to(device)
            total_len += l_target.shape[-1]
            lack_total_len += l_lack_labels.shape[-1]

            output = model(l_data)
            output = output.view([-1, output.shape[-1]])
            targets = l_target.view(-1)
            test_loss += F.cross_entropy(output, targets, ignore_index=-1).item()
            pred = output.argmax(1)
            pred = as_seaquence(pred.detach(), ahead=7)
            segmentation_correct += pred.eq(targets.view_as(pred)).sum().item()

            model.tatc.select_data_per_labels(l_data, pred, device)
            tatc_output = model.tatc()
            test_loss += F.cross_entropy(tatc_output, l_lack_labels.reshape(-1)).item()
            tatc_pred = tatc_output.argmax(1)
            
            print("true:", l_lack_labels[0])
            print("inference:", tatc_pred)
            lack_classifier_correct += tatc_pred.eq(l_lack_labels.view_as(tatc_pred)).sum().item()

            true_seq_labels += targets.view_as(pred).cpu().tolist()
            inf_seq_labels += pred.cpu().tolist()

            lack_labels_cpu = l_lack_labels.view_as(tatc_pred).cpu().tolist()
            tatc_pred_cpu = tatc_pred.cpu().tolist()
            true_finish_labels += lack_labels_cpu
            inf_finish_labels += tatc_pred_cpu
            inf_finish_proba += tatc_output[:, 1].view(-1).cpu().tolist()

            true_finish_labels_mat[i_batch] = lack_labels_cpu
            inf_finish_labels_mat[i_batch] = tatc_pred_cpu
            
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Seg Accuracy: {}/{} ({:.0f}%), lack Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss,
                  segmentation_correct, total_len, 100. * segmentation_correct / total_len,
                  lack_classifier_correct, lack_total_len, 100. * lack_classifier_correct / lack_total_len))

    print("seq f1:")
    print(precision_recall_fscore_support(true_seq_labels, inf_seq_labels))

    print("finish work:")
    print(precision_recall_fscore_support(true_finish_labels, inf_finish_labels))

    fpr, tpr, _ = roc_curve(true_finish_labels, inf_finish_proba)
    plt.plot(fpr, tpr)
    plt.savefig( Path(args.out_dir, 'finish_roc.png') )
    print("finish work AUC:")
    print(auc(fpr, tpr))
    
    for i in range(args.n_class -1):
        print("class {}:".format(i))
        print(precision_recall_fscore_support(true_finish_labels_mat[:, i], inf_finish_labels_mat[:, i]))
    
    print("低速:")
    print(precision_recall_fscore_support(true_finish_labels_mat[:5, :].ravel(), inf_finish_labels_mat[:5, :].ravel()))

    print("中速:")
    print(precision_recall_fscore_support(true_finish_labels_mat[5:10, :].ravel(), inf_finish_labels_mat[5:10, :].ravel()))

    print("高速:")
    print(precision_recall_fscore_support(true_finish_labels_mat[10:15, :].ravel(), inf_finish_labels_mat[10:15, :].ravel()))

    for i in range(5):
        start = 15+i*3
        end = 15+(i+1)*3
        print("作業{}中断再開:".format(i+1))
        print(precision_recall_fscore_support(true_finish_labels_mat[start:end, :].ravel(), inf_finish_labels_mat[start:end, :].ravel()))

    for i in range(5):
        start = 30+i*3
        end = 30+(i+1)*3
        print("作業{}中断:".format(i+1))
        print(precision_recall_fscore_support(true_finish_labels_mat[start:end, :].ravel(), inf_finish_labels_mat[start:end, :].ravel()))

    for i in range(5):
        start = 45+i*3
        end = 45+(i+1)*3
        print("作業{}欠損:".format(i+1))
        print(precision_recall_fscore_support(true_finish_labels_mat[start:end, :].ravel(), inf_finish_labels_mat[start:end, :].ravel()))    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/home/sh70k/mnt/tracker_data/test', help='path to dataset')
    parser.add_argument('--n-class', type=int, default=6, help='number of class')
    parser.add_argument('--test_seq-len', type=int, default=200, help='fixed seaquence length')
    parser.add_argument('--time-step', type=float, default=.25, help='fixed time interbal of input data')
    parser.add_argument('--test-data-file-pointer-path', default='./data/test_data_file_pointer', help='path to test data file pointer')
    parser.add_argument('--resume-model', default='/home/sh70k/mnt/tracker_data/results/model_ckpt_v1_average.pth', help='path to trained model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
    parser.add_argument('--out-dir', default='/home/sh70k/mnt/tracker_data/results', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
