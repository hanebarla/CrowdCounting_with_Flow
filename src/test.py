import torch
import torch.nn.functional as F
import torchvision
import argparse
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
    parser.add_argument('-nl', '--normarize_loss', type=bool, default=False)

    args = parser.parse_args()
    test_d_path = args.path
    model_weights = args.model_weight
    is_normalize = args.normarize_loss

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

    bar = Bar('testing... ', max=int(-(-data_len // minibatch_size)))
    for i, data in enumerate(TestLoader):
        inputs, persons, flows = data

        tm_img, t_img = inputs[0], inputs[1]
        t_person = persons[0]
        tm2t_flow = flows[0]

        tm_img, t_img = tm_img.to(device, dtype=torch.float),\
            t_img.to(device, dtype=torch.float)
        t_person = t_person.to(device, dtype=torch.float)
        tm2t_flow = tm2t_flow.to(device, dtype=torch.float)

        flow = torch.sum(tm2t_flow, dim=1)

        with torch.set_grad_enabled(False):
            output_before_forward = CANnet(tm_img, t_img)

            output = torch.sum(output_before_forward, dim=1, keepdim=True)

            # pixel range 0~1(float) → 0~255(float)
            if not is_normalize:
                output *= 255
                output = output.type(torch.uint8)
                output = output.type(torch.float)
                t_person *= 255
                t_person = t_person.type(torch.uint8)
                t_person = t_person.type(torch.float)

        d_mae = mae(output, t_person)
        d_mse = mse(output, t_person)
        d_rmse = torch.sqrt(d_mse)

        all_mae += d_mae.item()/int(-(-data_len // minibatch_size))
        all_rmse += d_rmse.item()/int(-(-data_len // minibatch_size))
        bar.next()
    bar.finish()
    print("MAE: {}, RMSE: {}".format(all_mae, all_rmse))


if __name__ == "__main__":
    test()
