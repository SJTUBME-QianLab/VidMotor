import argparse
import csv
import json
import os
import shutil
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from pathlib import Path
project_root = Path(__file__).resolve().parent
from VidMotor.loss_utils import counterfactual_loss, non_causal_loss
from VidMotor.feeder.feeder_STGCN import Feeder
from VidMotor.net.ST_GCN import CausalModel, NonCausalHead
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# Inline values from the selected ST-GCN fine-tuning YAML. Update `test_feeder_args` first; its `name` drives all output labels.
DATASET_CONFIG = {
    # Change these three values for a new task. `data_path` may be absolute.
    'name': 'SPHERE-Stair-Gait',
    'data_path': str(project_root / 'data' / 'Example' / 'data_and_label_SPHERE-Stair-Gait.pkl'),
    'time_process': {'overlength': 'downsampling_uniform_offset', 'underlength': 'upsampling_fill_zero'},
}
FIXED_SETTINGS = {
    'test_feeder_args': [DATASET_CONFIG],
    # `num_class` must match the labels.
    'model_args': {'in_channels': 3, 'num_class': 5, 'graph_args': {'layout': 'human36M', 'strategy': 'uniform'}},
    'model_nc_args': {'num_class': 5}, 'seed': 42, 'data_seed': 1439, 'num_epoch': 300, 'eval_interval': 1, 'num_worker': 0,
    'batch_size_grid': [4, 8], 'learning_rate_grid': [0.001, 0.0005, 0.0001], 'test_batch_size': 8,
    'weight_cf': 1.0, 'margin_cf': 0.6, 'weight_decay': 0.0005, 'data_centered': 7, 'data_dim': '3D',
    'data_norm': 'zscore', 'label_type': 'class_label', 'score_norm': False,
    'model_pretrain': str(project_root / 'VidMotor' / 'pretrain-ST-GCN.pt'),
}


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='VidMotor ST-GCN fine-tuning example')
    parser.add_argument('--device', type=int, default=[0], nargs='+', help='GPU indices')
    cli_args = parser.parse_args()
    return argparse.Namespace(device=cli_args.device, **FIXED_SETTINGS)


class Processor:
    def __init__(self, arg):
        self.arg = arg

    def load_model(self, ce_weights=None):
        output_device = self.arg.device[0]
        self.output_device = output_device
        self.model = CausalModel(**self.arg.model_args).cuda(output_device)
        self.nchead = NonCausalHead(**self.arg.model_nc_args).cuda(output_device)
        model_pretrain = torch.load(self.arg.model_pretrain, map_location='cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = {}
        for key, value in model_pretrain.items():
            new_key = key[7:] if key.startswith('module.') else key
            if new_key in model_dict and 'fcn' not in new_key:
                pretrained_dict[new_key] = value
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.loss = nn.CrossEntropyLoss(weight=ce_weights).cuda(output_device)
        if len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model, device_ids=self.arg.device, output_device=output_device)

    def load_optimizer(self):
        optimizer_args = dict(lr=self.arg.base_lr, momentum=0.9, nesterov=True, weight_decay=self.arg.weight_decay)
        self.optimizer_main = optim.SGD(self.model.parameters(), **optimizer_args)
        self.optimizer_nc = optim.SGD(self.nchead.parameters(), **optimizer_args)

    def train_one_epoch(self, train_loader):
        self.model.train()
        for data, label, _ in train_loader:
            if label.unique().numel() == 1:
                continue
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            output_c, feature_nc, output_cf = self.model(data, label)
            loss_c = self.loss(output_c, label)
            loss_cf = counterfactual_loss(output_cf, label, margin=self.arg.margin_cf)
            loss = loss_c + self.arg.weight_cf * loss_cf
            self.optimizer_main.zero_grad()
            loss.backward()
            self.optimizer_main.step()
            output_nc = self.nchead(feature_nc.detach())
            loss_nc = non_causal_loss(output_c.detach(), output_nc)
            self.optimizer_nc.zero_grad()
            loss_nc.backward()
            self.optimizer_nc.step()

    def evaluate(self, data_loader):
        self.model.eval()
        score_fragments = []
        with torch.no_grad():
            for data, _, _ in data_loader:
                data = data.float().cuda(self.output_device)
                output_c = self.model(data, label=None)
                score_fragments.append(output_c.cpu().numpy())
        output_score = np.concatenate(score_fragments)
        predicted = np.argmax(output_score, axis=1)
        true_label = np.asarray(data_loader.dataset.ori_label, dtype=int)
        probabilities = F.softmax(torch.from_numpy(output_score), dim=1).numpy()
        num_classes = self.arg.model_args['num_class']
        accuracy = data_loader.dataset.top_k(output_score, 1)
        average = 'binary' if num_classes == 2 else 'macro'
        f1 = f1_score(true_label, predicted, average=average, zero_division=0)
        balanced_acc = balanced_accuracy_score(true_label, predicted)
        try:
            if num_classes == 2:
                auc = roc_auc_score(true_label, probabilities[:, 1])
            else:
                auc = roc_auc_score(true_label, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes)))
        except ValueError:
            auc = float('nan')
        return {'accuracy': float(accuracy), 'balanced_acc': float(balanced_acc), 'f1': float(f1), 'auc': float(auc)}

    def fit(self, train_set, validation_loader, ce_weights, checkpoint_path):
        self.load_model(ce_weights)
        self.load_optimizer()
        train_loader = DataLoader(train_set, batch_size=self.arg.train_batch_size, shuffle=True, num_workers=self.arg.num_worker, drop_last=True)
        best_metrics = None
        best_epoch = None
        for epoch in range(self.arg.num_epoch):
            self.train_one_epoch(train_loader)
            should_evaluate = ((epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == self.arg.num_epoch)
            if not should_evaluate:
                continue
            validation_metrics = self.evaluate(validation_loader)
            if best_metrics is None or validation_metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = validation_metrics
                best_epoch = epoch + 1
                model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model_to_save.state_dict(), 'epoch': best_epoch, 'batch_size': self.arg.train_batch_size,
                            'learning_rate': self.arg.base_lr, 'validation_metrics': best_metrics}, checkpoint_path)
        return best_metrics, best_epoch

    def load_finetuned_checkpoint(self, checkpoint_path, ce_weights):
        self.load_model(ce_weights)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])


def prepare_datasets(arg, Feeder):
    """Create the original 80/20 split (fine-tune/validate); reuse the 20% set for the test example only for demonstration purposes.
    In practical scenarios, an additional held-out test set should be loaded for final performance evaluation."""
    dataset_config = arg.test_feeder_args[0]
    dataset = Feeder(dataset_config, arg.data_centered, arg.data_dim, arg.data_norm, arg.score_norm, arg.label_type)
    sample_names = dataset.sample_name
    # For this dataset, sample keys are `<subject>_...`. Change this extraction and the two matching expressions below for another naming rule.
    dataset_subjects = sorted({sample_name.split('_')[0] for sample_name in sample_names})
    random.seed(arg.data_seed)
    random.shuffle(dataset_subjects)
    num_subjects = len(dataset_subjects)
    num_train = int(0.8 * num_subjects)  # 80% training protocol
    train_subjects = dataset_subjects[:num_train]
    evaluation_subjects = dataset_subjects[num_train:]
    split_samples = {'train': [sample for sample in sample_names if any(subject + '_' in sample for subject in train_subjects)],
                     'evaluation': [sample for sample in sample_names if any(subject + '_' in sample for subject in evaluation_subjects)]}
    assert sum(len(names) for names in split_samples.values()) == len(sample_names)
    # Reuse the training-derived sequence length.
    common_args = (dataset_config, arg.data_centered, arg.data_dim, arg.data_norm, arg.score_norm, arg.label_type)
    train_set = Feeder(*common_args, data_length=None, data_indices=split_samples['train'])
    evaluation_set = Feeder(*common_args, data_length=train_set.data_length, data_indices=split_samples['evaluation'])
    num_classes = arg.model_args['num_class']
    class_counts = [train_set.label.count(class_id) for class_id in range(num_classes)]
    weights = [len(train_set.label) / count if count else 0 for count in class_counts]
    ce_weights = torch.tensor(weights)
    evaluation_loader = DataLoader(evaluation_set, batch_size=arg.test_batch_size, shuffle=False, num_workers=arg.num_worker, drop_last=False)
    print(f'Subject split: train={len(train_subjects)}, validation={len(evaluation_subjects)}')
    return train_set, evaluation_loader, ce_weights


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def named_metrics(metrics):
    return {'Accuracy': metrics['accuracy'], 'Balanced Accuracy': metrics['balanced_acc'], 'F1 Score': metrics['f1'], 'AUROC': metrics['auc']}


if __name__ == '__main__':
    arg = get_args()
    seed_torch(arg.seed)
    dataset_name = arg.test_feeder_args[0]['name']
    train_set, evaluation_loader, ce_weights = prepare_datasets(arg, Feeder)
    output_dir = project_root / 'example_results'
    checkpoint_dir = output_dir / 'grid_checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = list(product(arg.batch_size_grid, arg.learning_rate_grid))
    grid_rows = []

    for trial_index, (batch_size, learning_rate) in enumerate(grid, start=1):
        print(f'\nGrid trial {trial_index}/{len(grid)}: batch_size={batch_size}, learning_rate={learning_rate:g}')
        seed_torch(arg.seed)
        arg.train_batch_size = int(batch_size)
        arg.base_lr = float(learning_rate)
        checkpoint_path = checkpoint_dir / (f'trial_{trial_index:02d}_bs{batch_size}_lr{learning_rate:g}.pt')
        processor = Processor(arg)
        validation_metrics, best_epoch = processor.fit(train_set, evaluation_loader, ce_weights, checkpoint_path)
        grid_rows.append({'trial': trial_index, 'batch_size': batch_size, 'learning_rate': learning_rate, 'best_epoch': best_epoch,
                          'validation_accuracy': validation_metrics['accuracy'], 'validation_balanced_accuracy': validation_metrics['balanced_acc'],
                          'validation_f1': validation_metrics['f1'], 'validation_auc': validation_metrics['auc'], 'checkpoint': str(checkpoint_path)})
        m = named_metrics(validation_metrics)
        print(f'Best validation: Accuracy: {m["Accuracy"] * 100:.2f}% | Balanced Accuracy: {m["Balanced Accuracy"] * 100:.2f}% | '
              f'F1 Score: {m["F1 Score"] * 100:.2f}% | AUROC: {m["AUROC"] * 100:.2f}%')

    grid_csv_path = output_dir / f'{dataset_name}_grid_search.csv'
    grid_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['Trial', 'Batch Size', 'Learning Rate', 'Best Epoch', 'Accuracy', 'Balanced Accuracy', 'F1 Score', 'AUROC']
    with open(grid_csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in grid_rows:
            m = named_metrics({'accuracy': row['validation_accuracy'], 'balanced_acc': row['validation_balanced_accuracy'], 'f1': row['validation_f1'], 'auc': row['validation_auc']})
            writer.writerow({'Trial': row['trial'], 'Batch Size': row['batch_size'], 'Learning Rate': row['learning_rate'], 'Best Epoch': row['best_epoch'], **m})
    best_trial = max(grid_rows, key=lambda row: row['validation_accuracy'])
    best_model_path = output_dir / f'{dataset_name}_best_model.pt'
    shutil.copy2(best_trial['checkpoint'], best_model_path)
    arg.train_batch_size = int(best_trial['batch_size'])
    arg.base_lr = float(best_trial['learning_rate'])
    final_processor = Processor(arg)
    final_processor.load_finetuned_checkpoint(best_model_path, ce_weights)

    """Reuse the validation set for the test example only for demonstration purposes.
    In practical scenarios, an additional held-out test set should be loaded for final performance evaluation."""
    test_metrics = final_processor.evaluate(evaluation_loader)

    report = {
        'dataset': dataset_name,
        'split_protocol': '80% subject-wise training / 20% subject-wise evaluation',
        'selection_rule': ['Accuracy'],
        'best_hyperparameters': {
            'batch_size': best_trial['batch_size'],
            'learning_rate': best_trial['learning_rate'],
            'best_epoch': best_trial['best_epoch'],
        },
        'best_validation_metrics': named_metrics({
            'accuracy': best_trial['validation_accuracy'],
            'balanced_acc': best_trial['validation_balanced_accuracy'],
            'f1': best_trial['validation_f1'],
            'auc': best_trial['validation_auc'],
        }),
        'test_metrics': named_metrics(test_metrics),
        'best_model': str(best_model_path),
        'grid_summary': str(grid_csv_path),
    }
    report_path = output_dir / f'{dataset_name}_best_model_results.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2)

    for row in grid_rows:
        Path(row['checkpoint']).unlink(missing_ok=True)
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    m = named_metrics(test_metrics)
    print(f'\nBest metrics: Accuracy: {m["Accuracy"] * 100:.2f}% | Balanced Accuracy: {m["Balanced Accuracy"] * 100:.2f}% | '
          f'F1 Score: {m["F1 Score"] * 100:.2f}% | AUROC: {m["AUROC"] * 100:.2f}%')
    print(f'Best hyperparameters: Batch size={best_trial["batch_size"]}, Learning rate={best_trial["learning_rate"]:g}')
    print(f'Best model: {best_model_path}')
    print(f'Grid summary: {grid_csv_path}')
    print(f'Test report: {report_path}')
