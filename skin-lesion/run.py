import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import torchvision
import pandas as pd
import click


class MelanomaDataModule(pl.LightningDataModule):
    def __init__(self, fold, n_folds, batch_size, trainval_seed, test_seed):
        super().__init__()
        self.batch_size = batch_size
        self.fold = fold
        self.n_folds = n_folds
        self.trainval_seed = trainval_seed
        self.test_seed = test_seed

        self._load_data()

    def setup(self, stage):
        pass

    def _load_data(self):
        df = pd.read_csv("data/x_struc.csv", sep=";", decimal=',').drop(columns="Unnamed: 0")
        print('got columns', df.columns)
        images = torch.load('data/x_train_resized_normalized.pt')

        # train/test split
        idx = np.array(range(len(df)))
        trainval_idx = idx[df['subset'] == 'trainval']
        trainval_df = df[df['subset'] == 'trainval']
        test_idx = idx[df['subset'] != 'trainval']
        test_df = df[df['subset'] != 'trainval']

        img_order = {f.replace('.jpg', ''): i for i, f in enumerate(images['names'])}
        trainval_imgs = torch.stack([images['images'][img_order[i]] for i in trainval_df['image']])
        test_imgs = torch.stack([images['images'][img_order[i]] for i in test_df['image']])

        # get validation fold
        cv = StratifiedKFold(self.n_folds, random_state=self.trainval_seed, shuffle=True)
        splits = cv.split(np.zeros(len(trainval_df)), trainval_df['target'].values)
        for i, (train_idx, val_idx) in enumerate(splits):
            if i == self.fold:
                break
        else:
            raise RuntimeError('invalid fold index')

        # make datasets
        dropcol = ["target", "image", "patient_id", "subset"]
        self.patient_design = pd.get_dummies(trainval_df.iloc[train_idx]['patient_id']).values
        self.x_tr = torch.tensor(trainval_df.iloc[train_idx].drop(columns=dropcol).values).float()
        self.y_tr = torch.tensor(trainval_df.iloc[train_idx]["target"].values).float()
        self.train_idx = train_idx
        self.train_ds = TensorDataset(
            self.x_tr,
            trainval_imgs[train_idx],
            torch.tensor(self.patient_design).float(),
			self.y_tr,
        )

        self.val_idx = val_idx
        self.val_ds = TensorDataset(
            torch.tensor(trainval_df.iloc[val_idx].drop(columns=dropcol).values).float(),
            trainval_imgs[val_idx],
            torch.zeros(len(val_idx), self.patient_design.shape[1]).float(),
            torch.tensor(trainval_df.iloc[val_idx]["target"].values).float(),
        )

        self.test_idx = test_idx
        self.test_ds = TensorDataset(
            torch.tensor(test_df.drop(columns=dropcol).values).float(),
            test_imgs,
            torch.zeros(len(test_df), self.patient_design.shape[1]).float(),
            torch.tensor(test_df['target'].values).float(),
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=4, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=4, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, num_workers=4, batch_size=self.batch_size)


class SsrCnnClassificationModel(nn.Module):
    def __init__(self, x_struc, train_y, randef_design, randef_penalty):
        super().__init__()

		# unstructured predictor
        dpt = 0.3
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='valid'), nn.ReLU(), nn.Dropout(dpt),
            nn.Conv2d(16, 16, (3, 3), padding='valid'), nn.ReLU(), nn.Dropout(dpt),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding='valid'), nn.ReLU(), nn.Dropout(dpt),
            nn.Conv2d(32, 32, (3, 3), padding='valid'), nn.ReLU(), nn.Dropout(dpt),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(26912, 256), nn.Dropout(dpt), nn.ReLU(),
            nn.Linear(256, 64), nn.Dropout(dpt), nn.ReLU(),
            nn.Linear(64, 1)
        )

		# structured predictor
        self.use_structured = True
        ws = torch.linalg.inv(x_struc.T @ x_struc) @ x_struc.T @ train_y
        self.struc = nn.Linear(x_struc.shape[1], 1, bias = False)
        self.struc.weight = nn.Parameter(
            #ws.view(1, -1), requires_grad=True
            torch.zeros(1, x_struc.shape[1]), requires_grad=True
        )

		# random effect
        self.randef_penalty = randef_penalty
        self.randef_coef = nn.parameter.Parameter(
            torch.zeros(randef_design.shape[1], requires_grad=True)
        )

    def forward(self, x, u, r):
        if self.use_structured:
            fix = self.cnn(u) + self.struc(x)
            pred = fix.view(-1) + r @ self.randef_coef
        else:
            pred = self.cnn(u).view(-1)
        return pred

    def loss(self, pred, target):
        pos_weight = (1 + (1 - target).sum()) / (1 + target.sum())
        cls = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor([pos_weight]).to(pred.device)
        )
        rpen = self.randef_penalty * torch.mean(self.randef_coef**2)
        return cls + rpen


class Experiment(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, scheduler_gamma):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

    def forward(self, *args):
        if len(args) > 1:
            x, u, r, *_ = args
        else:
            x, u, r, *_ = args[0]

        return self.model(x, u, r)

    def training_step(self, batch, idx):
        x, u, r, t = batch
        p = self(x, u, r)
        l = self.model.loss(p, t)
        return l

    def validation_step(self, batch, idx):
        x, u, r, t = batch
        p = self(x, u, r)
        l = self.model.loss(p, t)
        return {'p': p, 't': t, 'l': l}

    @staticmethod
    def _aggregate_outputs(outputs):
        y_true, y_pred, loss = [], [], []
        for each in outputs:
            y_true.append(each['t'].cpu())
            y_pred.append(each['p'].cpu())
            loss.append(each['l'].cpu())
        return torch.cat(y_true).int(), torch.cat(y_pred), torch.tensor(loss)

    def validation_epoch_end(self, outputs):
        y_true, y_pred, loss = self._aggregate_outputs(outputs)
        loss = torch.mean(loss).item()
        acc = torch.mean((
            (y_true > 0.5) == (y_pred > 0.0)
        ).float())
        auc = roc_auc_score(y_true, y_pred)
        aps = average_precision_score(y_true, y_pred)

        self.log('val_auc', auc)
        self.log('val_loss', loss)
        self.log('val_aps', aps)
        tqdm.write(f'epoch {self.current_epoch} | validation | auc: {auc:.4f} - accuracy: {acc:.4f} - AUPRC: {aps:.4f} - loss: {loss:.4f}')

    def configure_optimizers(self):
        optims = [optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )]
        scheds = [optim.lr_scheduler.ExponentialLR(
            optims[0], gamma=self.scheduler_gamma
        )]

        return optims, scheds


@click.group()
def main():
    pass


@main.command()
@click.argument('fold', envvar='SLURM_ARRAY_TASK_ID', type=int)
@click.argument('n_folds', envvar='SLURM_ARRAY_TASK_COUNT', type=int)
@click.option('--randef-penalty', default=1.0/400, type=float)
def train(fold, n_folds, randef_penalty):
    ckp_fname = f'checkpoints/testpreds_fold={fold}_pen={randef_penalty:.0e}.pt'
    if os.path.exists(ckp_fname):
        print('checkpoint exists, quitting', ckp_fname)
        return

    tqdm.write(f'\n\n*** TRAINING FOLD {fold} OF {n_folds} ***\n')
    tqdm.write(f'using penalty {randef_penalty}\n')
    data = MelanomaDataModule(
        fold, n_folds, batch_size=1024, trainval_seed=29875325, test_seed=36535761
    )
    model = SsrCnnClassificationModel(
        data.x_tr, data.y_tr,
        randef_design=data.patient_design,
        randef_penalty=randef_penalty,
    )
    experiment = Experiment(model, learning_rate=0.001, weight_decay=0.00001, scheduler_gamma=0.98)

    ckpkb = ModelCheckpoint(
        filename='fold=%d-pen=%.0e-{epoch}-{val_aps:.4f}' % (fold, randef_penalty),
        monitor='val_aps', save_top_k=1, mode='max'
    )
    trainer = pl.Trainer(
        gpus=-1, max_epochs=1000, gradient_clip_val=10, enable_progress_bar=False,
        callbacks=[
            ckpkb, EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=True),
        ],
    )
    trainer.fit(experiment, data)

    print('loading checkpoint', ckpkb.best_model_path)
    model.use_structured = False
    experiment.load_from_checkpoint(ckpkb.best_model_path)
    test_preds = np.concatenate(trainer.predict(experiment, datamodule=data))
    torch.save({
        'randef_penalty': randef_penalty,
        'fold': fold,
        'predictions': test_preds,
        'indices': data.test_idx,
        'structured_weights': experiment.model.struc.weight.detach(),
        'random_effects': experiment.model.randef_coef.detach(),
    }, ckp_fname)
    print('saved to', ckp_fname)


@main.command()
def aggregate():
    print('reading data')
    preds = {}
    weights = {}
    idx = None
    for fname in os.listdir('checkpoints'):
        if not fname.endswith('.pt') or not fname.startswith('testpreds'):
            continue
        print(fname)

        parts = fname.replace('.pt', '').split('_')
        penalty = parts[2].split('=')[1]
        if penalty not in preds:
            preds[penalty] = []
        if penalty not in weights:
            weights[penalty] = []

        sd = torch.load(os.path.join('checkpoints', fname))
        if idx is None:
            idx = sd['indices']
        assert np.all(idx == sd['indices'])
        preds[penalty].append(sd['predictions'])
        weights[penalty].append(sd['structured_weights'])

    print('got networks', {k: len(v) for k, v in preds.items()})

    print('saving')
    dfs = []
    for k, v in preds.items():
        df = pd.DataFrame([x.ravel() for x in v]).T
        df.columns = [f'network_{i}' for i in range(len(v))]
        df['idx'] = idx
        df['penalty'] = k
        dfs.append(df)
    pd.concat(dfs).reset_index().to_csv('ensemble_preds_heldout_nost.csv', index=False)

    dfs = []
    for k, v in weights.items():
        df = pd.DataFrame([x.ravel() for x in v]).T
        df.columns = [f'network_{i}' for i in range(len(v))]
        df['penalty'] = k
        dfs.append(df)
    pd.concat(dfs).reset_index().to_csv('ensemble_weights_penalized_heldout_nost.csv', index=False)


if __name__ == '__main__':
    main()
