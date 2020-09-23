import torch
import torch.nn.functional as F
import torchvision
import argparse
import matplotlib.pyplot as plt
from progress.bar import Bar
from utils import model
from utils import functions
from utils import load_datasets as LD


def test():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('-p', '--path', default='TestData_Path.csv')  # Testdata path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-mw', '--model_weight', default="CrowdCounting_model_cpu_epoch_50.pth")

    args = parser.parse_args()
    test_d_path = args.path
    model_weights = args.model_weight

    minibatch_size = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CANnet = model.CANNet()
    CANnet.to(device)
    CANnet.load_state_dict(torch.load(model_weights), strict=False)
    CANnet.eval()

    # Test Data Loader Settings
    trans = torchvision.transforms.ToTensor()
    Testdataset = LD.CrowdDatasets(
        transform=trans,
        width=args.width,
        height=args.height,
        Trainpath=test_d_path,
        test_on=True)
    TestLoader = torch.utils.data.DataLoader(
        Testdataset, batch_size=minibatch_size, shuffle=False)
    data_len = len(Testdataset)

    # Loss Func
    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    all_mae = 0
    all_rmse = 0

    bar = Bar('testing... ', max=int(data_len / minibatch_size) + 1)
    for i, data in enumerate(TestLoader):
        inputs, persons, flows = data

        tm_img, t_img = inputs[0], inputs[1]
        t_person = persons[0]
        tm2t_flow = flows[0]

        tm_img, t_img = tm_img.to(device, dtype=torch.float),\
            t_img.to(device, dtype=torch.float)
        t_person = t_person.to(device, dtype=torch.float)
        tm2t_flow = tm2t_flow.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            output_before_forward = CANnet(tm_img, t_img)

        output = torch.sum(output_before_forward, dim=1, keepdim=True)

        d_mae = mae(output, t_person)
        d_mse = mse(output, t_person)
        d_rmse = torch.sqrt(d_mse)

        all_mae += d_mae
        all_rmse += d_rmse
        bar.next()
    bar.finish()
    print("MAE: {}, RMSE: {}".format(all_mae, all_rmse))


if __name__ == "__main__":
    test()
