import csv
import os

import cv2
import numpy as np
import torch
import torchvision

ras2bits = 0.71
IP = {0: 202.5, 1: 247.5, 2: 292.5, 3: 157.5, 5: 337.5, 6: 22.5, 7: 67.5, 8: 112.5}


class CrowdDatasets(torch.utils.data.Dataset):
    def __init__(self, transform=None, width=1280, height=720, Trainpath="Data/TrainData_Path.csv", test_on=False):
        super().__init__()
        self.transform = transform
        self.width = width
        self.height = height
        self.out_width = int(width / 8)
        self.out_height = int(height / 8)
        self.test_on = test_on
        with open(Trainpath) as f:
            reader = csv.reader(f)
            self.Pathes = [row for row in reader]

    def __len__(self):
        return len(self.Pathes)

    def __getitem__(self, index):
        """
        CSV Pathlist reference
        -------
            train
                index 0: input image(step t),
                index 1: person label(step t),
                index 2: input label(step t-1),
                index 3: person label(step t-1),
                index 4: label flow(step t-1 2 t),
                index 5: input image(step t+1),
                index 6: preson label(step t+1),
                index 7: label flow(step t 2 t+1)

            test
                index 0: input image(step tm),
                index 1: person label(step tm),
                index 2: input image(step t),
                index 3: person label(step t),
                index 4: label flow(step tm 2 t)
        """
        if self.test_on:
            test_pathlist = self.Pathes[index]
            tm_img_path = test_pathlist[0]
            tm_person_path = test_pathlist[1]
            t_img_path = test_pathlist[2]
            t_person_path = test_pathlist[3]
            tm2t_flow_path = test_pathlist[4]

            t_input, t_person = self.gt_img_density(t_img_path, tm_img_path)
            tm_input, tm_person = self.gt_img_density(tm_img_path, tm_person_path)
            tm2t_flow = self.gt_flow(tm2t_flow_path)

            return [tm_input, t_input], [t_person], [tm2t_flow]

        else:
            pathlist = self.Pathes[index]
            t_img_path = pathlist[0]
            t_person_path = pathlist[1]
            t_m_img_path = pathlist[2]
            t_m_person_path = pathlist[3]
            t_m_t_flow_path = pathlist[4]
            t_p_img_path = pathlist[5]
            t_p_person_path = pathlist[6]
            t_t_p_flow_path = pathlist[7]

            t_input, t_person = self.gt_img_density(t_img_path, t_person_path)
            tm_input, tm_person = self.gt_img_density(t_m_img_path, t_m_person_path)
            tp_input, tp_person = self.gt_img_density(t_p_img_path, t_p_person_path)
            tm2t_flow = self.gt_flow(t_m_t_flow_path)
            t2tp_flow = self.gt_flow(t_t_p_flow_path)

            return [tm_input, t_input, tp_input], [tm_person, t_person, tp_person], [tm2t_flow, t2tp_flow]

    def IndexProgress(self, i, gt_flow_edge, h, s):
        oheight = self.out_height
        owidth = self.out_width
        if i == 4:
            grid_i = np.zeros((oheight, owidth, 1))
            return grid_i
        elif i == 9:
            gt_flow_edge_ndarr = np.array(gt_flow_edge)
            gtflow_sum = np.sum(gt_flow_edge_ndarr, axis=0)
            grid_i = gtflow_sum
            return grid_i
        else:
            grid_i = np.where((h >= IP[i] * ras2bits) & (h < ((IP[i] + 45) % 360) * ras2bits), 1, 0)
            grid_i = np.array(grid_i, dtype=np.uint8)
            grid_i = s * grid_i
            grid_i = cv2.resize(grid_i, (owidth, oheight))  # width, height
            grid_i_inner = grid_i[1:(oheight-1), 1:(owidth-1)]
            grid_i_edge = grid_i
            grid_i_inner = np.pad(grid_i_inner, 1)
            grid_i_edge[1:(oheight-1), 1:(owidth-1)] = 0
            grid_i_inner = np.reshape(grid_i_inner, (oheight, owidth, 1))  # height, width, channel
            grid_i_edge = np.reshape(grid_i_edge, (oheight, owidth, 1))
            gt_flow_edge.append(grid_i_edge)

            return grid_i_inner

    def gt_flow(self, path):

        if not os.path.isfile(path):
            return print("No such file: {}".format(path))

        gt_flow_list = []
        gt_flow_edge = []
        img = cv2.imread(path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(img_hsv)
        for i in range(10):
            grid_i = self.IndexProgress(i, gt_flow_edge, h, s)
            gt_flow_list.append(grid_i)

        gt_flow_img_data = np.concatenate(gt_flow_list, axis=2)
        gt_flow_img_data /= 255

        gt_flow_img_data = self.transform(gt_flow_img_data)

        return gt_flow_img_data

    def gt_img_density(self, img_path, mask_path):

        if not os.path.isfile(img_path):
            return print("No such file: {}".format(img_path))
        if not os.path.isfile(mask_path):
            return print("No such file: {}".format(mask_path))

        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (self.width, self.height))  # width, height
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_path, 0)
        if mask_img is None:
            return print("CRC error: {}".format(mask_path))
        mask_img = np.reshape(mask_img, (mask_img.shape[0], mask_img.shape[1], 1))

        input_img = input_img / 255
        mask_img = cv2.resize(mask_img, (self.out_width, self.out_height)) / 255  # width, height

        input_img = self.transform(input_img)
        mask_img = self.transform(mask_img)

        return input_img, mask_img


if __name__ == "__main__":
    trans = torchvision.transforms.ToTensor()
    Trains = CrowdDatasets(transform=trans, Trainpath="CrowdCounting_with_Flow/Data/TrainData_Path.csv")
    inputs, persons, flows = Trains[0]
    inputs_tm, inputs_t, inputs_tp = inputs[0].size(), inputs[1].size(), inputs[2].size()
    person_tm, person_t, person_tp = persons[0].size(), persons[1].size(), persons[2].size()
    flow_tm2t, flow_t2tp = flows[0].size(), flows[1].size()

    T_all = [inputs_tm, inputs_t, inputs_tp, person_tm, person_t, person_tp, flow_tm2t, flow_t2tp]

    print("\ntm_img shape: {}, t_img shape: {}, tp_img shape: {}".format(inputs_tm, inputs_t, inputs_tp))
    print("tm_person shape: {}, t_person shape: {}, tp_person shape: {}".format(person_tm, person_t, person_tp))
    print("tm2t_flow shape: {}, t2tp_flow shape: {} \n".format(flow_tm2t, flow_t2tp))

    data_all_num = len(Trains)
    print("Data nums: {}\n".format(data_all_num))

    correct = 0
    wrong = 0

    for data in Trains:
        d_inputs, d_persons, d_flows = data
        d_all = [d_inputs[0].size(), d_inputs[1].size(), d_inputs[2].size(),
                 d_persons[0].size(), d_persons[1].size(), d_persons[2].size(),
                 d_flows[0].size(), d_flows[1].size()]

        if bool(T_all == d_all):
            correct += 1
        else:
            wrong += 1

    print("Correct Data num: {} / {}".format(correct, data_all_num))
    print("wrong Data num: {} / {}".format(wrong, data_all_num))
