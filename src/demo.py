import torch
import torch.nn.functional as F
import torchvision
import argparse
from progress.bar import Bar
from utils import model
from utils import functions
from utils import load_datasets as LD


default_model_path = "models/2020-11-09_h_360_w_640_lr_0.0004719475861107414_wd_0.007307616924875147_e_100.pth"


def demo():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('-p', '--path', default='TestData_Path.csv')  # Testdata path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-mw', '--model_weight', default=default_model_path)

    args = parser.parse_args()
    test_d_path = args.path
    model_weights = args.model_weight

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CANnet = model.CANNet()
    CANnet.to(device)
    CANnet.load_state_dict(torch.load(model_weights), strict=False)
    CANnet.eval()

    trans = torchvision.transforms.ToTensor()
    Testdataset = LD.CrowdDatasets(
        transform=trans,
        width=args.width,
        height=args.height,
        Trainpath=test_d_path,
        test_on=True)
    TestLoader = torch.utils.data.DataLoader(
        Testdataset, batch_size=1, shuffle=True)

    for i, data in enumerate(TestLoader):
        if i > 0:
            break
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

            functions.output_to_img(tm_img[0, :, :, :], output_before_forward[0, :, :, :])
            functions.NormalizeQuiver(tm_img[0, :, :, :], output_before_forward[0, :, :, :])


if __name__ == "__main__":
    demo()
