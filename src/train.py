import os
import numpy as np
import random
import torch
import torch.optim as optim
import torchvision
import argparse
import matplotlib.pyplot as plt
from progress.bar import Bar
from utils import model
from utils import functions
from utils import load_datasets as LD
# import pytorch_memlab as PM


def train():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'TrainData_Path.csv'
                                                 """)
    parser.add_argument('-p', '--path', default='TrainData_Path.csv')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-wd', '--width', type=int, default=640)
    parser.add_argument('-ht', '--height', type=int, default=360)
    args = parser.parse_args()
    train_d_path = args.path

    minibatch_size = 48
    epock_num = args.epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    CANnet = model.CANNet()
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        CANnet = torch.nn.DataParallel(CANnet)

    """
    for p in CANnet.parameters():
        p.data.clamp_(-0.1, 0.1)
    """

    CANnet.to(device)
    CANnet.train()

    # reporter = PM.MemReporter(CANnet)

    torch.backends.cudnn.benchmark = True

    trans = torchvision.transforms.ToTensor()
    Traindataset = LD.CrowdDatasets(
        transform=trans,
        width=args.width,
        height=args.height,
        Trainpath=train_d_path)

    TrainLoader = torch.utils.data.DataLoader(
        Traindataset, batch_size=minibatch_size, shuffle=True, num_workers=8)
    data_len = len(Traindataset)

    criterion = functions.AllLoss(batchsize=minibatch_size, optical_loss_on=0)
    """
    optimizer = optim.SGD(
        CANnet.parameters(),
        lr=0.01,
        weight_decay=0.001)
    """
    optimizer = optim.Adam(
        CANnet.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01)

    # reporter.report()

    losses = []

    for epock in range(epock_num):
        e_loss = 0.0
        e_floss = 0.0
        e_closs = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epock + 1, epock_num))
        print('-------------')
        print('（train）')

        bar = Bar('training... ', max=int(-(-data_len // minibatch_size)))

        for i, data in enumerate(TrainLoader):

            torch.cuda.empty_cache()
            inputs, persons, flows = data

            tm_img, t_img, tp_img = inputs[0], inputs[1], inputs[2]
            tm_person, t_person, tp_person = persons[0], persons[1], persons[2]
            tm2t_flow, t2tp_flow = flows[0], flows[1]

            tm_img, t_img, tp_img = tm_img.to(device, dtype=torch.float),\
                t_img.to(device, dtype=torch.float),\
                tp_img.to(device, dtype=torch.float)

            tm_person, t_person, tp_person = tm_person.to(
                device, dtype=torch.float), t_person.to(
                device, dtype=torch.float), tp_person.to(
                device, dtype=torch.float)

            tm2t_flow, t2tp_flow = tm2t_flow.to(device, dtype=torch.float),\
                t2tp_flow.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                output_befoer_forward = CANnet(tm_img, t_img)
                output_before_back = CANnet(t_img, tm_img)
                output_after_back = CANnet(tp_img, t_img)

            with torch.set_grad_enabled(True):
                output_after_forward = CANnet(t_img, tp_img)

            loss, floss, closs = criterion.forward(tm_person, t_person, tm2t_flow,
                                                   output_befoer_forward, output_before_back,
                                                   output_after_forward, output_after_back)

            e_loss += loss.item() / int(-(-data_len // minibatch_size))
            e_floss += floss.item() / int(-(-data_len // minibatch_size))
            e_closs += closs.item() / int(-(-data_len // minibatch_size))

            assert not torch.isnan(floss).item(), "Loss Error: is nan !!"

            loss.backward()
            optimizer.step()
            bar.next()
            print(" Floss: {:8f}, Closs: {:8f}".format(floss.item(), closs.item()))

            del tm_img, t_img, tp_img
            del tm_person, t_person, tp_person
            del tm2t_flow, t2tp_flow
        bar.finish()

        losses.append(e_loss)
        print('-------------')
        print('epoch {} || Epoch_Loss:{}, Epoch_FlowLoss:{}, Epock_CycleLoss:{}'.format(epock + 1, e_loss, e_floss, e_closs))
        if (epock + 1) == (epock_num - 5) or (epock + 1) == epock_num or (epock+1) % 100 == 0:
            save_path = os.path.join("models", 'CrowdCounting_{}_{}_epoch_{}.pth'.format(args.height, args.width, epock + 1))
            torch.save(CANnet.state_dict(), save_path)

    print("Training Done!!")
    # reporter.report()
    # CANnet = CANnet.to('cpu')
    # print("Save Done!!")

    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title("loss")
    plt.show()
    plt.savefig("loss_record.png")


if __name__ == "__main__":
    # シード値の固定
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train()
