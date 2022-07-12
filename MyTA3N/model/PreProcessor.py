from torch import nn
import torch

from dataset import TSNDataSetModified as TSNDataSet
from opts import parser

import pandas as pd
from torch.utils.data import DataLoader


class PreProcessor(nn.Module):
    def __init__(self, input_size=2048, num_segments: int = 5, domains=2, use_classifier=False,transform=True):
        super(PreProcessor, self).__init__()
        self.use_classifier = use_classifier
        self.is_train = True
        self.num_segments = num_segments

        # Fc -> ReLu Fc
        self.domain_predictor = nn.Sequential(nn.Linear(in_features=input_size * num_segments, out_features=input_size),
                                              nn.ReLU(),
                                              nn.Linear(in_features=input_size, out_features=1))

    def _transformSourceVector(self, x: torch.Tensor):

        x = torch.concat((x, x, torch.zeros(x.size())), dim=-1)

        return x

    def _transformTargetVector(self, x: torch.Tensor):

        x = torch.concat((x, torch.zeros(x.size()), x), dim=-1)

        return x

    def forward(self, x_source: torch.Tensor, x_target: torch.Tensor = None):
        """
        Transform the given tensors and make prediction about its procedence (source or target).
        """
        # Distinguish the case in which we are dealing with training
        segment_size: int = x_source.size()[1]
        source_batch_size: int = x_source.size()[0]

        # Re-size for doing the classification
        x_source_fc = x_source.view(source_batch_size, -1)

        if self.is_train:
            assert x_target is not None
            target_batch_size: int = x_target.size()[0]
            x_target_fc = x_target.view(target_batch_size, -1)
            # Concatenate
            concat: torch.Tensor = torch.concat((x_source_fc, x_target_fc), dim=0)
            # Transform the source and target vectors
            x_source: torch.Tensor = self._transformSourceVector(x_source)
            x_target: torch.Tensor = self._transformTargetVector(x_target)

            # Do prediction
            domain_pred = self.domain_predictor(concat)

        else:
            assert x_target is None
            # Do prediction
            domain_pred = self.domain_predictor(x_source_fc)

            # In case of not using the classifier do as if it was the target vector
            prediction = torch.sigmoid(domain_pred) if self.use_classifier else 0

            if prediction > 0.5:
                # Source set
                x_source = self._transformSourceVector(x_source)
            else:
                x_source = self._transformTargetVector(x_source)

        return x_source, x_target, domain_pred

    def train(self, mode: bool = True):
        super(PreProcessor, self).train()
        self.is_train = mode

    def eval(self):
        super(PreProcessor, self).eval()
        self.is_train = False


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == '__main__':
    global args, best_prec1, writer_train, writer_val
    # Parse the arguments
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre_processor = PreProcessor(input_size=2048).to(device)

    if args.modality == 'Audio' or 'RGB' or args.modality == 'ALL':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source &
    # target are same
    num_source = len(pd.read_pickle(args.train_source_data + ".pkl"))
    num_target = len(pd.read_pickle(args.train_target_data + ".pkl"))

    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter * args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    source_set = TSNDataSet(args.train_source_data + ".pkl", args.train_source_list, num_dataload=num_source_train,
                            num_segments=args.num_segments,
                            new_length=1, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            )

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False,
                                                sampler=source_sampler, num_workers=args.workers, pin_memory=True)

    target_set = TSNDataSet(args.train_target_data + ".pkl", args.train_target_list, num_dataload=num_target_train,
                            num_segments=args.num_segments,
                            new_length=1, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            )

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False,
                                                sampler=target_sampler, num_workers=args.workers, pin_memory=True)

    ## Retrieve test set
    test_dataset_path: str = "test".join(args.train_target_data.rsplit("train", 1))
    test_list_dataset_path: str = "test".join(args.train_target_list.rsplit("train", 1))
    target_test_set = TSNDataSet(test_dataset_path + ".pkl", test_list_dataset_path, num_dataload=num_target_train,
                                 num_segments=args.num_segments,
                                 new_length=1, modality=args.modality,
                                 image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                 "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                                 random_shift=False,
                                 test_mode=True,
                                 )
    target_test_sampler = torch.utils.data.sampler.RandomSampler(target_test_set)
    target_test_set_loader = torch.utils.data.DataLoader(target_test_set, batch_size=args.batch_size[2], shuffle=False,
                                                         sampler=target_test_sampler, num_workers=args.workers,
                                                         pin_memory=True)

    test_dataset_path: str = "test".join(args.train_source_data.rsplit("train", 1))
    test_list_dataset_path: str = "test".join(args.train_source_list.rsplit("train", 1))
    source_test_set = TSNDataSet(test_dataset_path + ".pkl", test_list_dataset_path, num_dataload=num_target_train,
                                 num_segments=args.num_segments,
                                 new_length=1, modality=args.modality,
                                 image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                 "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                                 random_shift=False,
                                 test_mode=True,
                                 )
    source_test_sampler = torch.utils.data.sampler.RandomSampler(source_test_set)
    source_test_set_loader = torch.utils.data.DataLoader(source_test_set, batch_size=args.batch_size[2], shuffle=False,
                                                         sampler=source_test_sampler, num_workers=args.workers,
                                                         pin_memory=True)

    data_loader = enumerate(zip(source_loader, target_loader))

    attn_epoch_source = torch.Tensor()
    attn_epoch_target = torch.Tensor()
    pre_processor.train()

    criterion_processor = nn.BCEWithLogitsLoss()
    optimizer_processor = torch.optim.Adam(pre_processor.parameters(), lr=0.01)

    for e in range(args.epochs):
        j = 0
        epoch_loss = 0
        epoch_acc = 0
        data_loader = enumerate(zip(source_loader, target_loader))
        for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_loader:
            # Pre-Train the pre-processor
            x = torch.concat((source_data, source_data, torch.zeros(source_data.size())), dim=-1)
            source_data, target_data = source_data.to("cuda"), target_data.to("cuda")
            optimizer_processor.zero_grad()
            # Predict label
            _, _, pred_domain = pre_processor(source_data, target_data)

            # Create ground-truth labels
            domain_labels = torch.concat((torch.ones(source_data.size()[0]), torch.zeros(target_data.size()[0])), dim=0) \
                .view(source_data.size()[0] + target_data.size()[0], ).to(device)
            pred_domain = pred_domain.view(source_data.size()[0] + target_data.size()[0], )
            loss_domain = criterion_processor(pred_domain, domain_labels)
            loss_domain.backward()
            optimizer_processor.step()
            acc = binary_acc(pred_domain, domain_labels)

            j += 1

            epoch_loss += loss_domain.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / j:.5f} | Acc: {epoch_acc / j:.3f}')

    # Now do validation

    data_test_loader = enumerate(zip(source_test_set_loader, target_test_set_loader))

    for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_test_loader:
        # Assign labels
        pass
