import time
import torch
import torch.optim as optim
import torchvision
import argparse
from progress.bar import Bar
from utils import model
from utils import functions
from utils import load_datasets as LD
# import pytorch_memlab as PM


def train():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TrainData_Path.csv'
                                                 """)
    parser.add_argument('-p', '--path', default='Data/TrainData_Path.csv')
    parser.add_argument('-wd', '--width', type=int, default=1280)
    parser.add_argument('-ht', '--height', type=int, default=720)
    args = parser.parse_args()
    train_d_path = args.path

    minibatch_size = 32
    epock_num = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    CANnet = model.CANNet()
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        CANnet = torch.nn.DataParallel(CANnet)
    CANnet.to(device)
    CANnet.train()

    # reporter = PM.MemReporter(CANnet)

    torch.backends.cudnn.benchmark = True

    trans = torchvision.transforms.ToTensor()
    Traindataset = LD.CrowdDatasets(transform=trans, width=args.width, height=args.height, Trainpath=train_d_path)
    TrainLoader = torch.utils.data.DataLoader(Traindataset, batch_size=minibatch_size, shuffle=True)
    data_len = len(Traindataset)

    criterion = functions.AllLoss(batchsize=minibatch_size)
    # optimizer = optim.Adam(CANnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.9)
    optimizer = optim.SGD(CANnet.parameters(), lr=0.01, weight_decay=0.5)

    # reporter.report()

    for epock in range(epock_num):
        e_loss = 0.0

        e_time_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epock, epock_num))
        print('-------------')
        print('（train）')

        bar = Bar('training... ', max=int(data_len/minibatch_size)+1)

        for i, data in enumerate(TrainLoader):

            inputs, persons, flows = data

            tm_img, t_img, tp_img = inputs[0], inputs[1], inputs[2]
            tm_person, t_person, tp_person = persons[0], persons[1], persons[2]
            tm2t_flow, t2tp_flow = flows[0], flows[1]

            tm_img, t_img, tp_img = tm_img.to(device, dtype=torch.float),\
                t_img.to(device, dtype=torch.float),\
                tp_img.to(device, dtype=torch.float)

            tm_person, t_person, tp_person = tm_person.to(device, dtype=torch.float), \
                t_person.to(device, dtype=torch.float), \
                tp_person.to(device, dtype=torch.float)

            tm2t_flow, t2tp_flow = tm2t_flow.to(device, dtype=torch.float),\
                t2tp_flow.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                output_befoer_forward = CANnet(tm_img, t_img)
                output_after_forward = CANnet(t_img, tp_img)
                output_before_back = CANnet(t_img, tm_img)

            with torch.set_grad_enabled(True):
                output_after_back = CANnet(tp_img, t_img)

            loss = criterion(tm_person, t_person, tm2t_flow,
                             output_befoer_forward, output_before_back,
                             output_after_forward, output_after_back)

            e_loss += loss.item()/int(data_len/minibatch_size)
            loss.backward()
            optimizer.step()
            bar.next()

        e_time_stop = time.time()
        e_time = e_time_stop - e_time_start
        bar.finish()

        print('-------------')
        print('epoch {} || Epoch_Loss:{}'.format(epock, e_loss))
        print('timer:  {:.4f} sec.'.format(e_time))
        if (epock+1) == (epock_num-5) or (epock+1) == epock_num:
            torch.save(CANnet.state_dict(), 'CrowdCounting_model_cpu_epoch_{}.pth'.format(epock+1))

    print("Training Done!!")
    # reporter.report()
    CANnet = CANnet.to('cpu')
    print("Save Done!!")


if __name__ == "__main__":
    train()
