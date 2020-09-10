import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from utils import model
from utils import functions
from utils import load_datasets as LD


def train():
    minibatch_size = 1
    epock_num = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CANnet = model.CANNet()
    CANnet.to(device)
    CANnet.train()

    torch.backends.cudnn.benchmark = True

    trans = torchvision.transforms.ToTensor()
    Traindataset = LD.CrowdDatasets(transform=trans, Trainpath="CrowdCounting_with_Flow/Data/TrainData_Path.csv")
    TrainLoader = torch.utils.data.DataLoader(Traindataset, batch_size=minibatch_size, shuffle=True)

    criterion = functions.AllLoss()
    optimizer = optim.Adam(CANnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    for epock in range(epock_num):
        e_loss = 0.0

        e_time_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epock, epock_num))
        print('-------------')
        print('（train）')

        for i, data in enumerate(TrainLoader):

            inputs, persons, flows = data

            tm_img, t_img, tp_img = inputs[0], inputs[1], inputs[2]
            tm_person, t_person, tp_person = persons[0], persons[1], persons[2]
            tm2t_flow, t2tp_flow = flows[0], flows[1]

            tm_img, t_img, tp_img = Variable(tm_img).to(device, dtype=torch.float),\
                Variable(t_img).to(device, dtype=torch.float),\
                Variable(tp_img).to(device, dtype=torch.float)

            tm_person, t_person, tp_person = Variable(tm_person).to(device, dtype=torch.float), \
                Variable(t_person).to(device, dtype=torch.float), \
                Variable(tp_person).to(device, dtype=torch.float)

            tm2t_flow, t2tp_flow = Variable(tm2t_flow).to(device, dtype=torch.float),\
                Variable(t2tp_flow).to(device, dtype=torch.float)

            optimizer.zero_grad()

            output_befoer_forward = CANnet(tm_img, t_img)
            output_after_forward = CANnet(t_img, tp_img)
            output_before_back = CANnet(t_img, tm_img)
            output_after_back = CANnet(tp_img, t_img)

            loss = criterion(tm_person, t_person, tm2t_flow,
                             output_befoer_forward, output_before_back,
                             output_after_forward, output_after_back)

            e_loss += loss.item()
            loss.backend()
            optimizer.step()

        e_time_stop = time.time()
        e_time = e_time_start - e_time_stop

        print('-------------')
        print('epoch {} || Epoch_Loss:{:.4f}'.format(epock, e_loss / minibatch_size))
        print('timer:  {:.4f} sec.'.format(e_time))

        print("Training Done!!")


if __name__ == "__main__":
    train()
