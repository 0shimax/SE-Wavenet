import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from data.data_loader import ActivDataset, loader
from models.ete_waveform import EteWave
from models.post_process import as_seaquence
from optimizer.radam import RAdam


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def main(args):
    model = EteWave(args.n_class).to(device)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    # setup optimizer
    optimizer = RAdam(model.parameters())

    train_data_file_names =\
        [line.rstrip() for line in open(args.train_data_file_pointer_path)]
    test_data_file_names =\
        [line.rstrip() for line in open(args.test_data_file_pointer_path)]

    train_dataset = ActivDataset(train_data_file_names, args.root_dir,
                                 seq_len=args.train_seq_len, time_step=args.time_step,
                                 is_train=True)
    test_dataset = ActivDataset(test_data_file_names, args.root_dir,
                                seq_len=args.test_seq_len, time_step=args.time_step,
                                is_train=False, test_in_train=True)
    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    test(args, model, test_loader)


def l1_loss(model, reg=1e-4):
    loss = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            loss += reg * torch.sum(torch.abs(param))
    return loss


def train(args, model, optimizer, data_loader):
    model.train()
    for epoch in range(args.epochs):
        for i, (l_data, l_target, l_lack_labels) in enumerate(data_loader):
            l_data = l_data.to(device)
            l_target = l_target.to(device)
            l_lack_labels = l_lack_labels.to(device)
            # _, in_ch, _ = l_data.shape

            model.zero_grad()
            optimizer.zero_grad()

            # output of shape (seq_len, batch, num_directions * hidden_size)
            output = model(l_data)
            output = output.reshape([-1, args.n_class])
            targets = l_target.view(-1)
            series_loss = F.cross_entropy(output,
                                          targets,
                                          ignore_index=-1)
            with torch.no_grad():
                N_series_loss = series_loss.detach().mean() + 3*series_loss.detach().std()
            series_loss = series_loss.mean()

            inf_labels = output.argmax(1)
            model.tatc.select_data_per_labels(l_data, inf_labels, device)
            # tatc out shape is (n_non_zero_labels*n_batch, 2)
            tatc_output = model.tatc()
            tatc_loss = F.cross_entropy(tatc_output,
                                        l_lack_labels.reshape(-1),
                                        ignore_index=-1)
            with torch.no_grad():
                N_tatc_loss = tatc_loss.detach().mean() + 3*tatc_loss.detach().std()
            tatc_loss = tatc_loss.mean()
            if N_tatc_loss > N_series_loss:
                loss = series_loss + N_tatc_loss/N_series_loss*tatc_loss
            else:
                loss = N_series_loss/N_tatc_loss*series_loss + tatc_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                       '{}/model_ckpt.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
               '{}/model_ckpt.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_len = 0
    with torch.no_grad():
        for i_batch, (l_data, l_target, l_lack_labels) in enumerate(data_loader):
            l_data = l_data.to(device)
            l_target = l_target.to(device)
            l_lack_labels = l_lack_labels.to(device)
            total_len += l_target.shape[-1]

            output = model(l_data)
            output = output.view([-1, output.shape[-1]])
            targets = l_target.view(-1)
            test_loss += F.cross_entropy(output, targets, ignore_index=-1).item()

            pred = output.argmax(1)
            model.tatc.select_data_per_labels(l_data, pred, device)
            tatc_output = model.tatc()
            test_loss += F.cross_entropy(tatc_output, l_lack_labels.reshape(-1)).item()

            pred = as_seaquence(pred.detach(), ahead=7)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            for p, t in zip(pred, targets):
                print(p, t)
            print(l_lack_labels)
            print(tatc_output.argmax(1))

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, total_len, 100. * correct / total_len))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data/train', help='path to dataset')
    parser.add_argument('--n-class', type=int, default=6, help='number of class')
    parser.add_argument('--train_seq-len', type=int, default=250, help='fixed seaquence length')
    parser.add_argument('--test_seq-len', type=int, default=200, help='fixed seaquence length')
    parser.add_argument('--time-step', type=float, default=.25, help='fixed time interbal of input data')
    parser.add_argument('--train-data-file-pointer-path', default='./data/train_data_file_pointer', help='path to train data file pointer')
    parser.add_argument('--test-data-file-pointer-path', default='./data/train_data_file_pointer', help='path to test data file pointer')
    parser.add_argument('--resume-model', default='./results/_tatc_ckpt.pth', help='path to trained model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch-size', type=int, default=12, help='input batch size')  # seq_len=200 -> 12,
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--out-dir', default='./results', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
