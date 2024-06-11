import torch
import numpy as np
from sklearn.metrics import f1_score
import random
from tqdm import tqdm
from src.trainer.metrics import Evaluate

metric = Evaluate()

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, summary_writer, device, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.summary_writer = summary_writer
        self.device = device
        self.local_rank = local_rank
        self.n_updates = 0

    def save_snapshot(self, model, epoch, step):
        snapshot = {}
        if self.args.dist_train:
            snapshot["model_state"] = model.module.state_dict()
        else:
            snapshot["model_state"] = model.state_dict()
        snapshot["epoch_run"] = epoch
        snapshot['step_count'] = step
        torch.save(snapshot, self.args.save_dir + f'/model/model_repeat{self.args.repeat}.pt')

    def _forward_epoch(self, model, batched_data):
        graphs, pk_values = batched_data
        graphs = graphs.to(self.device)
        pk_values = pk_values.to(self.device)
        results = model(graphs)

        return results, pk_values

    def _eval(self, model, batched_data):
        graphs, pk_values = batched_data
        graphs = graphs.to(self.device)
        pk_values = pk_values.to(self.device)
        preds = model(graphs)

        return preds.flatten(), pk_values

    def train_iter(self, model, batched_data):
        self.optimizer.zero_grad()
        results, true_pk = self._forward_epoch(model, batched_data)
        loss = self.loss_fn(results.flatten(), true_pk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.max_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss


    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        train_loss = []
        for batch_idx, batched_data in enumerate(train_loader):
            loss = self.train_iter(model, batched_data)
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Train/loss', train_loss, epoch_idx)

        return train_loss


    def val(self, model, val_loader):
        model.eval()
        preds, true_pks = [], []
        with torch.no_grad():
            for batch_idx, batched_data in enumerate(val_loader):
                pred, true_pk = self._eval(model, batched_data)
                preds.append(pred.cpu())
                true_pks.append(true_pk.cpu())
            preds = torch.cat(preds)
            true_pks = torch.cat(true_pks)
        loss = self.loss_fn(preds, true_pks)
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Val/loss', loss.item(), self.n_updates)
        return loss



    def fit(self, model, train_loader, val_loader, test_loader):
        best_loss = torch.tensor(float('inf'))
        count = 0
        for epoch in tqdm(range(self.args.epoch_run, self.args.n_epoch)):
            train_loss = self.train_epoch(model, train_loader, epoch)
            if epoch < self.args.warmup_epoch:
                if epoch % 10 == 1:
                    self.save_snapshot(model, epoch, step=self.lr_scheduler._step_count)
                continue
            val_loss = self.val(model, val_loader)
            # preds, true_pks = self.predict(model, test_loader)
            # rmse, mae, r, sd = metric.evaluate(preds, true_pks)
            # print(round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4))
            if self.args.dist_train:
                if self.local_rank == 0:
                    print('Epoch {}: Train loss {:.4f} || Val loss {:.4f}'.format(epoch, train_loss, val_loss))
            else:
                print('Epoch {}: Train loss {:.4f} || Val loss {:.4f}'.format(epoch, train_loss, val_loss))

            if best_loss > val_loss:
                best_loss = val_loss
                count = 0
                self.save_snapshot(model, epoch, step=self.lr_scheduler._step_count)
            elif (abs(best_loss - val_loss) < 0.1) and (train_loss > 0.001):
                count = 0
            else:
                count += 1

            if (count == self.args.patience) | (epoch == self.args.n_epoch - 1) | (train_loss < 0.0001):
                # print('Early stopping!')
                best_weights = torch.load(self.args.save_dir + f'/model/model_repeat{self.args.repeat}.pt', map_location=self.args.device)
                torch.save(best_weights, self.args.save_dir + f'/model/best_model_repeat{self.args.repeat}.pt')
                try:
                    model.load_state_dict(best_weights['model_state'])
                except:
                    model.module.load_state_dict(best_weights['model_state'])

                break

    def predict(self, model, test_loader):
        model.eval()
        preds, true_pks = [], []
        with torch.no_grad():
            for batch_idx, batched_data in enumerate(test_loader):
                self.optimizer.zero_grad()
                pred, true_pk = self._eval(model, batched_data)
                preds.append(pred.cpu())
                true_pks.append(true_pk.cpu())
            preds = torch.cat(preds)
            true_pks = torch.cat(true_pks)

        return preds.numpy(), true_pks.numpy()

