import torch


class GANLoss:
    def __init__(self, device: torch.device):
        self.device = device

        self.adv_loss = torch.nn.BCELoss()
        ce_weight = torch.tensor([6.38, 3.05, 1.09, 1.0, 2.25], device=device)
        self.aux_loss = torch.nn.CrossEntropyLoss(weight=ce_weight)

        self.adv_loss.to(device)
        self.aux_loss.to(device)

    def loss_function(self, pred_validity, real_validity, pred_label, real_label):
        adv_loss = self.adv_loss(pred_validity, real_validity)
        aux_loss = self.aux_loss(pred_label, real_label)
        loss = (adv_loss + aux_loss) / 2
        return loss
