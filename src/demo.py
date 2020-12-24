import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import argparse
from progress.bar import Bar
from utils import model
from utils import loss_function as Losses
from utils import val_function as Valid
from utils import load_datasets as LD


normal_path = "models/2020-11-09_h_360_w_640_lr_0.0004719475861107414_wd_0.007307616924875147_e_100.pth"
direct_path = "models/2020-12-14_h_360_w_640_lr_0.0022961891160849187_wd_3.776888134461094e-07_e_100.pth"


def demo():
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('-p', '--path', default='TestData_Path.csv')  # Testdata path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-nw', '--normal_weight', default=normal_path)
    parser.add_argument('-dw', '--direct_weight', default=direct_path)
    parser.add_argument('-num', '--img_num', default=10)

    args = parser.parse_args()
    test_d_path = args.path
    normal_weights = args.normal_weight
    direct_weights = args.direct_weight
    num = args.img_num

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CANnet = model.CANNet()
    CANnet.to(device)
    CANnet.load_state_dict(torch.load(normal_weights), strict=False)
    CANnet.eval()

    D_CANnet = model.CANNet()
    D_CANnet.to(device)
    D_CANnet.load_state_dict(torch.load(direct_weights), strict=False)
    D_CANnet.eval()

    trans = torchvision.transforms.ToTensor()
    Testdataset = LD.CrowdDatasets(
        transform=trans,
        width=args.width,
        height=args.height,
        Trainpath=test_d_path,
        test_on=True)
    TestLoader = torch.utils.data.DataLoader(
        Testdataset, batch_size=1, shuffle=True)
    sigma = torch.nn.Sigmoid()

    img_dict_keys = ['input',
                     'label',
                     'normal',
                     'normal_quiver',
                     'normal_dense_res',
                     'direct',
                     'direct_quiver',
                     'direct_dense_res']

    img_dict = {
        img_dict_keys[0]: ('img', None),
        img_dict_keys[1]: ('img', None),
        img_dict_keys[2]: ('img', None),
        img_dict_keys[3]: ('quiver', None),
        img_dict_keys[4]: ('img', None),
        img_dict_keys[5]: ('img', None),
        img_dict_keys[6]: ('quiver', None),
        img_dict_keys[7]: ('img', None)
    }

    DemoImg = Valid.CompareOutput(img_dict_keys)

    for i, data in enumerate(TestLoader):
        if i >= num:
            print("\n")
            break
        inputs, persons, flows = data

        tm_img, t_img = inputs[0], inputs[1]
        tm_person = persons[0]
        tm2t_flow = flows[0]

        tm_img, t_img = tm_img.to(device, dtype=torch.float),\
            t_img.to(device, dtype=torch.float)
        tm_person = tm_person.to(device, dtype=torch.float)
        tm2t_flow = tm2t_flow.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            output_normal = CANnet(tm_img, t_img)
            # output_normal = sigma(output_normal) - 0.5

            output_direct = D_CANnet(tm_img, t_img)
            # output_direct = sigma(output_direct) - 0.5

        input_num = tm_img[0, :, :, :].detach().cpu().numpy()
        input_num = input_num.transpose((1, 2, 0))
        label_num = tm_person[0, :, :, :].detach().cpu().numpy()
        label_num = label_num.transpose((1, 2, 0))

        normal_num = output_normal[0, :, :, :].detach().cpu().numpy()
        normal_quiver = Valid.NormalizeQuiver(normal_num)
        normal_num = normal_num.transpose((1, 2, 0))

        direct_num = output_direct[0, :, :, :].detach().cpu().numpy()
        direct_quiver = Valid.NormalizeQuiver(direct_num)
        direct_num = direct_num.transpose((1, 2, 0))

        normal_dense = Valid.tm_output_to_dense(normal_num)
        direct_dense = Valid.tm_output_to_dense(direct_num)

        normal_res_dense = Valid.output_res_img(np.squeeze(label_num), normal_dense)
        direct_res_dense = Valid.output_res_img(np.squeeze(label_num), direct_dense)

        img_dict = {
            img_dict_keys[0]: ('img', input_num),
            img_dict_keys[1]: ('img', label_num),
            img_dict_keys[2]: ('img', normal_dense),
            img_dict_keys[3]: ('quiver', normal_quiver),
            img_dict_keys[4]: ('img', normal_res_dense),
            img_dict_keys[5]: ('img', direct_dense),
            img_dict_keys[6]: ('quiver', direct_quiver),
            img_dict_keys[7]: ('img', direct_res_dense),
        }

        DemoImg.append_pred(img_dict)

        print("{} / {} done\r".format((i+1), num), end="")

    DemoImg.plot_img()
    DemoImg.save_fig()


if __name__ == "__main__":
    demo()
