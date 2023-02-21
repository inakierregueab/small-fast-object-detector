import torch
from tqdm import tqdm


class DetectorEngine:
    def __init__(self, model):
        self.model = model

    def configure_optimizers(self, optimizer_config):
        self.optimizer = None

    def configure_data_loaders(self, loaders_config):
        self.train_loader = None
        self.val_loader = None

    def configure_loss_fn(self, loss_config):
        self.loss_fn = None

    def forward(self, train=True):
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            data = self.data.to(self.device)
            labels = self.labels.to(self.device)

            # Initialize results dict
            result = {}

            model_out = self.model(data)

    def train_loop(self, epoch):
        print(f"Training epoch {epoch}/{self.epochs}")

        nbs = 64  # nominal batch size
        batch_size = len(next(iter(self.train_loader))[0])
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        last_opt_step = -1

        loop = tqdm(self.train_loader)
        avg_batches_loss = 0
        loss_epoch = 0
        nb = len(self.train_loader)
        self.optimizer.zero_grad()

        for idx, (images, bboxes) in enumerate(loop):
            images = images.float() / 255
            # images = images.to(config.DEVICE, non_blocking=True)
            # BBOXES AND CLASSES ARE PUSHED to.(DEVICE) INSIDE THE LOSS_FN

            # Forward pass
            # with torch.cuda.amp.autocast():
            out = self.model(images)
            loss = self.loss_fn(out, bboxes, pred_size=images.shape[2:4], batch_idx=idx, epoch=epoch)
            avg_batches_loss += loss
            loss_epoch += loss

            # Backward pass

    def train(self, train_config):
        self.epochs = train_config['epochs']


        self.model.train()

        for epoch in range(self.epochs):
            pass

