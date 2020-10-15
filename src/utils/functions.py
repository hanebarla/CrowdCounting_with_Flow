import os
import cv2
import numpy as np
import torch
# import model
# import pytorch_memlab
# from utils  import model


class AllLoss():
    def __init__(self, optical_loss_on=1, batchsize=1):
        # super().__init__()
        self.optical_loss_on = optical_loss_on
        if self.optical_loss_on == 0:
            print("***Not Using Optical Loss***")
        self.bathsize = batchsize

    def forward(self, tm_personlabel, t_person_label, tm2t_flow_label,
                output_before_foward, output_before_back,
                output_aftter_foward, output_after_back, alpha=1, beta=1e-4):

        floss = self.flow_loss(output_before_forward=output_before_foward,
                               output_after_forward=output_aftter_foward,
                               label=t_person_label)

        closs = self.cycle_loss(output_before_foward=output_before_foward,
                                output_before_back=output_before_back,
                                output_after_foward=output_aftter_foward,
                                output_after_back=output_after_back)

        loss_combi = floss + alpha * closs

        if self.optical_loss_on == 1:
            oloss = self.optical_loss(
                tm_personlabel,
                tm2t_flow_label,
                output_before_foward)
            loss_combi += beta * oloss

        # loss_combi /= self.bathsize

        """
        floss_nan = (torch.sum(torch.isnan(floss)).item() == 0)
        closs_nan = (torch.sum(torch.isnan(closs)).item() == 0)
        oloss_nan = (torch.sum(torch.isnan(oloss)).item() == 0)

        print("floss_nan: {}, closs_nan: {}, oloss_nan: {}".format(floss_nan, closs_nan, oloss_nan))
        """
        return loss_combi, floss, closs  # 後で減らす

    def flow_loss(self, output_before_forward, output_after_forward, label):
        est_sum_before = self.sum_flow(output_before_forward)
        est_sum_after = torch.sum(output_after_forward, dim=1, keepdim=True)
        # print("{}, {}".format(torch.max(est_sum_after), torch.max(est_sum_before)))

        res_before = label - est_sum_before
        res_after = label - est_sum_after

        # print(res_before)

        se_before = res_before * res_before
        se_after = res_after * res_after

        floss = torch.sum((se_before + se_after)) / self.bathsize

        return floss

    def cycle_loss(self, output_before_foward, output_before_back,
                   output_after_foward, output_after_back):

        res_before = output_before_foward - self.back_flow(output_before_back)
        res_after = output_after_foward - self.back_flow(output_after_back)

        se_before = torch.sum(
            (res_before * res_before),
            dim=1,
            keepdim=True) / self.bathsize
        se_after = torch.sum(
            (res_after * res_after),
            dim=1,
            keepdim=True) / self.bathsize

        closs = torch.sum((se_before + se_after)) / self.bathsize

        return closs

    def optical_loss(self, before_person_label, flow_label, flow_output):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        indisize = before_person_label.size()
        indicator = torch.where(
            before_person_label > 0.9,
            torch.ones(
                indisize[0],
                indisize[1],
                indisize[2],
                indisize[3]).to(device),
            torch.zeros(
                indisize[0],
                indisize[1],
                indisize[2],
                indisize[3]).to(device))
        se = (flow_label - flow_output) * \
            (flow_label - flow_output) / self.bathsize

        loss = torch.sum((indicator * se)) / self.bathsize

        return loss

    def sum_flow(self, output):
        """
        Sum tm2tflow to trajectories map
        -------------
            Slide flows to sum flows for trajectories map
            Step t-1 2 t flow -> Step t Trajectory map
            output(10 channel) -> Trajectories map(1 channel)
        """
        o_shape = output.size()

        for i in range(10):
            if i == 4 or i == 9:
                continue
            elif i == 0:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (-1, -1), dims=(1, 2))
                output[:, i, :o_shape[2], (o_shape[3]-1)] = 0.0
                output[:, i, (o_shape[2]-1), :o_shape[2]] = 0.0
            elif i == 1:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], -1, dims=1)
                output[:, i, o_shape[2]-1, :o_shape[3]] = 0.0
            elif i == 2:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (-1, 1), dims=(1, 2))
                output[:, i, :o_shape[2], 0] = 0.0
                output[:, i, (o_shape[2]-1), :o_shape[2]] = 0.0
            elif i == 3:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], -1, dims=2)
                output[:, i, :o_shape[2], o_shape[3]-1] = 0.0
            elif i == 5:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], 1, dims=2)
                output[:, i, :o_shape[2], 0] = 0.0
            elif i == 6:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (1, -1), dims=(1, 2))
                output[:, i, 0, :(o_shape[3])] = 0.0
                output[:, i, :o_shape[2], o_shape[3]-1] = 0.0
            elif i == 7:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], 1, dims=1)
                output[:, i, 0, :o_shape[3]] = 0.0
            elif i == 8:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (1, 1), dims=(1, 2))
                output[:, i, 0, :o_shape[3]] = 0.0
                output[:, i, :o_shape[2], 0] = 0.0

        return torch.sum(output, dim=1, keepdim=True)

    def back_flow(self, output):
        o_shape = output.size()
        output[:, 9, :, :] = 0.0

        for i in range(10):
            if i == 4 or i == 9:
                continue
            elif i == 0:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (-1, -1), dims=(1, 2))
                output[:, 9, :o_shape[2], (o_shape[3]-1)] += output[:, i, :o_shape[2], (o_shape[3]-1)]
                output[:, 9, (o_shape[2]-1), :o_shape[2]] += output[:, i, (o_shape[2]-1), :o_shape[2]]
            elif i == 1:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], -1, dims=1)
                output[:, 9, o_shape[2]-1, :o_shape[3]] += output[:, i, o_shape[2]-1, :o_shape[3]]
            elif i == 2:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (-1, 1), dims=(1, 2))
                output[:, 9, :o_shape[2], 0] += output[:, i, :o_shape[2], 0]
                output[:, 9, (o_shape[2]-1), :o_shape[2]] += output[:, i, (o_shape[2]-1), :o_shape[2]]
            elif i == 3:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], -1, dims=2)
                output[:, 9, :o_shape[2], o_shape[3]-1] += output[:, i, :o_shape[2], o_shape[3]-1]
            elif i == 5:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], 1, dims=2)
                output[:, 9, :o_shape[2], 0] += output[:, i, :o_shape[2], 0]
            elif i == 6:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (1, -1), dims=(1, 2))
                output[:, 9, 0, :o_shape[3]] += output[:, i, 0, :(o_shape[3])]
                output[:, 9, :o_shape[2], o_shape[3]-1] = output[:, i, :o_shape[2], o_shape[3]-1]
            elif i == 7:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], 1, dims=1)
                output[:, 9, 0, :o_shape[3]] += output[:, i, 0, :o_shape[3]]
            elif i == 8:
                output[:, i, :, :] = torch.roll(output[:, i, :, :], (1, 1), dims=(1, 2))
                output[:, 9, 0, :o_shape[3]] += output[:, i, 0, :o_shape[3]]
                output[:, 9, :o_shape[2], 0] += output[:, i, :o_shape[2], 0]

        return output


def output_to_img(output):
    root = os.getcwd()
    imgfolder = os.path.join(root, "images/")

    # d_max = torch.max(output).item()
    output_num = output.detach().cpu().numpy()
    out = output_num[0, 0, :, :]

    heatmap = cv2.resize(out, (out.shape[1]*8, out.shape[0]*8))
    heatmap = np.array(heatmap*(255), dtype=np.uint8)

    cv2.imwrite(imgfolder+"test.png", heatmap)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    can_model = model.CANNet(load_weights=True)
    can_model.to(device)
    # reporter = pytorch_memlab.MemReporter(can_model)

    criterion = AllLoss()

    x1 = torch.ones(1, 3, 720, 1280).to(device)
    x2 = torch.ones(1, 3, 720, 1280).to(device)
    x3 = torch.ones(1, 3, 720, 1280).to(device)

    tm_person = torch.zeros(1, 1, 90, 160).to(device)
    t_person = torch.zeros(1, 1, 90, 160).to(device)
    tm2t_flow = torch.zeros(1, 10, 90, 160).to(device)

    with torch.set_grad_enabled(False):
        output_befoer_forward = can_model(x1, x2)
        output_after_forward = can_model(x2, x3)
        output_before_back = can_model(x2, x1)

    with torch.set_grad_enabled(True):
        output_after_back = can_model(x3, x2)

    loss = criterion.forward(tm_person, t_person, tm2t_flow,
                             output_befoer_forward, output_before_back,
                             output_after_forward, output_after_back)
    loss.backward()
    # reporter.report()

    e_loss = loss.item()
    print("loss: {}".format(e_loss))
