import torch
import numpy as np


class AllLoss(torch.nn.Module):
    def __init__(self, optical_loss_on=1):
        super().__init__()
        self.optical_loss_on = optical_loss_on

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
            oloss = self.optical_loss(tm_personlabel, tm2t_flow_label, output_before_foward)
            loss_combi += beta * oloss

        return loss_combi

    def flow_loss(self, output_before_forward, output_after_forward, label):
        est_sum_before = torch.sum(output_before_forward, dim=-1)
        est_sum_after = torch.sum(output_after_forward, dim=-1)

        # サイズ判定を行う
        res_before = label - est_sum_before
        res_after = label - est_sum_after

        se_before = torch.mm(res_before, res_before)
        se_after = torch.mm(res_after, res_after)

        floss = torch.sum((se_before + se_after))

        return floss

    def cycle_loss(self, output_before_foward, output_before_back,
                   output_after_foward, output_after_back):

        res_before = output_before_foward - output_before_back
        res_after = output_after_foward - output_after_back

        se_before = torch.sum(torch.mm(res_before, res_before), dim=-1)
        se_after = torch.sum(torch.mm(res_after, res_after), dim=-1)

        closs = torch.sum((se_before + se_after))

        return closs

    def optical_loss(self, before_person_label, flow_label, flow_output):
        indicator = np.where(before_person_label > 0.9, 1, 0)
        se = (flow_label - flow_output) * (flow_label - flow_output)

        loss = se.sum(indicator * se)

        return loss
