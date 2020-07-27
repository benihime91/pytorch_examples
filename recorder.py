class Recoder:
    """
    Records train_loss, validation_loss, validation_metrics,
    learning_rate, momentum
    """

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.errs = []
        self.lrs = []
        self.moms = []

    def update_loss_metrics(self, loss: float, val_loss: float, acc: float, error: float):
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        self.accs.append(acc)
        self.errs.append(error)

    def update_lr_moms(self, lr: float, mom: float):
        self.lrs.append(lr)
        self.moms.append(mom)

    def reset(self):
        """Reset the recorded values"""
        self.losses = []
        self.val_losses = []
        self.errs = []
        self.accs = []
        self.lrs = []
        self.moms = []
