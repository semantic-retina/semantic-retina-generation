import torch


class ACGANLoss:
    def __init__(self, device: torch.device):
        self.device = device

        self.adv_loss = torch.nn.BCEWithLogitsLoss()
        ce_weight = torch.tensor([6.38, 3.05, 1.09, 1.0, 2.25], device=device)
        self.aux_loss = torch.nn.NLLLoss()

        self.adv_loss.to(device)
        self.aux_loss.to(device)

    def loss_function(self, pred_validity, real_validity, pred_label, real_label):
        adv_loss = self.adv_loss(pred_validity, real_validity)
        # aux_loss = self.aux_loss(pred_label, real_label)
        # loss = (adv_loss + aux_loss) / 2
        loss = adv_loss
        return loss
