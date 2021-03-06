import torch
import torch.nn.functional as F
import torchvision
import argparse
from progress.bar import Bar
from utils import model
from utils import loss_function as Losses
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

    minibatch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        Testdataset, batch_size=minibatch_size, shuffle=False, num_workers=8)
    data_len = len(Testdataset)
    batch_repeet_num = int(-(-data_len // minibatch_size))

    # Loss Func
    criterion = Losses.AllLoss(device=device, batchsize=minibatch_size, optical_loss_on=0)

    all_mae = 0
    all_rmse = 0

    bar = Bar('testing... ', max=batch_repeet_num)
    for i, data in enumerate(TestLoader):
        torch.cuda.empty_cache()
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

            output = criterion.sum_flow(output_before_forward)

            # pixel range 0~1(float) → 0~255(float)
            if not is_normalize:
                output *= 255
                output = output.type(torch.uint8)
                output = output.type(torch.float)
                t_person *= 255
                t_person = t_person.type(torch.uint8)
                t_person = t_person.type(torch.float)

            abs_err = torch.abs(output - t_person)
            root_squ_err = torch.sqrt(abs_err * abs_err)

            d_mae = torch.sum(abs_err) / minibatch_size
            d_rmse = torch.sum(root_squ_err) / minibatch_size

            all_mae += d_mae.item()/batch_repeet_num
            all_rmse += d_rmse.item()/batch_repeet_num

        bar.next()
        del tm_img, t_img
        del t_person
        del tm2t_flow
    bar.finish()
    print("MAE: {}, RMSE: {}".format(all_mae, all_rmse))


if __name__ == "__main__":
    test()
