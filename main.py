import torch
import torch.nn.functional as F
from utils import CitationDataset, MMD
import os.path as osp
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from models import BernNet
import argparse

def class_accuracies(y, pred):
    correct = (pred == y)
    class_correct = {}
    class_total = {}
    misclassified_info = {}
    misclassified_counts = {}
    for i in range(y.max()+1):
        class_mask = (y == i)
        class_correct[i] = correct[class_mask].sum().item()
        class_total[i] = class_mask.sum().item()
        misclassified_indices = np.where((pred != y).cpu().numpy() & class_mask.cpu().numpy())[0]
        if misclassified_indices.size > 0:
            misclassified_classes = pred[misclassified_indices].cpu().numpy()
            count = np.bincount(misclassified_classes, minlength=y.max()+1)
            misclassified_counts[i] = [round(num, 2) for num in (count / class_total[i]).tolist()]
    class_acc = {i: round(class_correct[i] / class_total[i], 4) if class_total[i] > 0 else 0 for i in class_correct}
    return class_acc, class_total, misclassified_counts


def weighted_entropy_minimization_loss(output):
    probs = F.softmax(output, dim=1)
    log_probs = F.log_softmax(output, dim=1)

    a = torch.sum(probs, dim=0)
    class_weights = a / torch.sum(a)
    entropy_loss = -torch.sum(probs * log_probs / class_weights, dim=1).mean()
    return entropy_loss

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of DGSDA')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--hidden', type=int, default=128, help='hidden_dim')
    parser.add_argument('--K', type=int, default=8, help='polynomial order')
    parser.add_argument('--dropout_ratio', type=float, default=0.3, help='dropout ratio1')
    parser.add_argument('--dp_ratio', type=float, default=0.3, help='dropout ratio2')
    parser.add_argument('--epoch', type=int, default=100, help='train epochs')
    parser.add_argument('--source', type=str, default='ACMv9', help='source domain data')
    parser.add_argument('--target', type=str, default='Citationv1', help='target domain data')
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha weight')
    parser.add_argument('--beta', type=float, default=0.5, help='beta weight')
    parser.add_argument('--gamma', type=float, default=0.05, help='gamma weight')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index, default is 0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    hidden = args.hidden
    lr = args.lr
    weight_decay = args.wd
    source_path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)
    source_dataset = CitationDataset(source_path, args.source)

    target_path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.target)
    target_dataset = CitationDataset(target_path, args.target)

    print(source_dataset[0])
    print(target_dataset[0])

    model = BernNet(source_dataset.num_node_features, hidden, source_dataset.num_classes, args.dropout_ratio, args.dp_ratio, args.K)
    source_data = source_dataset[0]
    target_data = target_dataset[0]

    if torch.cuda.is_available():
        model = model.cuda()
        source_data = source_data.cuda()
        target_data = target_data.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(source_data)
        loss = F.cross_entropy(output[source_data.train_mask], source_data.y[source_data.train_mask])

        theta_s = model.prop1.temp
        theta_t = model.prop2.temp

        theta_loss = F.l1_loss(theta_s, theta_t)

        source_feature = F.relu(model.lin1(source_data.x))
        target_feature = F.relu(model.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)

        target_outputs = model(target_data, False)

        weight_enloss = weighted_entropy_minimization_loss(target_outputs)

        total_loss = loss + theta_loss*args.alpha + mmd_loss*args.beta + weight_enloss*args.gamma

        total_loss.backward()
        optimizer.step()

        model.eval()
        output = model(source_data)
        val_loss = F.cross_entropy(output[source_data.val_mask], source_data.y[source_data.val_mask])
        target_output = model(target_data, False)
        t_val_loss = F.cross_entropy(target_output, target_data.y)
        pred = target_output.max(1)[1]
        acc = pred.eq(target_data.y).sum().item() / len(target_data.y)
        print(f"epoch:{epoch}   Val_loss:{val_loss:.4f}"
              f" t_Val_loss:{t_val_loss:.4f} acc:{acc:.4f}")

    model.eval()
    theta_1 = torch.relu(model.prop1.temp.detach().cpu()).numpy()
    print('Theta_1:', [float('{:.4f}'.format(i)) for i in theta_1])
    theta_2 = torch.relu(model.prop2.temp.detach().cpu()).numpy()
    print('Theta_2:', [float('{:.4f}'.format(i)) for i in theta_2])
    theta_3 = torch.relu(model.prop3.temp.detach().cpu()).numpy()
    print('Theta_3:', [float('{:.4f}'.format(i)) for i in theta_3])
    _, pred = model(source_data).max(dim=1)
    correct = pred[source_data.test_mask].eq(source_data.y[source_data.test_mask]).sum()
    acc = correct / int(source_data.test_mask.sum())
    print("source accuracy= {:.4f}".format(acc))
    class_acc, class_total, total_error = class_accuracies(source_data.y[source_data.test_mask], pred[source_data.test_mask])
    print("Class-wise Accuracies:", class_acc)
    print("Class-wise Node Counts:", class_total)
    print("Class-wise Error:", total_error)

    print('----------------------------------------------------------------------')
    model.eval()
    _, pred = model(target_data, False).max(dim=1)
    correct = pred.eq(target_data.y).sum()
    acc2 = correct / len(target_data.y)
    print("target accuracy= {:.4f}".format(acc2))
    class_acc, class_total, total_error = class_accuracies(target_data.y, pred)
    print("Class-wise Accuracies:", class_acc)
    print("Class-wise Node Counts:", class_total)
    print("Class-wise Error:", total_error)


