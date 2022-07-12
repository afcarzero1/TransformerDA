import torch.utils.data as data

import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import pickle
import pandas as pd
from colorama import init
from colorama import Fore, Back, Style

init(autoreset=True)


class VideoRecord(object):
    """
    This is a class for holding the information about a video record.

    This class is made for the EPIC-KITCHENS-DATASET. It holds the information about a clip of an action.

    """

    def __init__(self, i, row, num_segments):
        """
        Initializer of the class
            Args:
                i(int) : It
                row(pd.Series) : Contains all information about the clip. Start timestamp, stop timestamp, verb class,
                    noun class
                num_segments(int): It is the number of the segment to which it belongs
        """
        self._data: pd.Series = row
        self._index: int = i
        self._seg: int = num_segments

    @property
    def segment_id(self):
        return self._data.uid #fixme : I substituted the

    @property
    def path(self):
        return self._index

    @property
    def num_frames(self):
        return int(self._seg)  # self._data[1])

    @property
    def label(self):
        if ("verb_class" in self._data):# and ("noun_class" in self._data):
            return int(self._data.verb_class)#, int(self._data.noun_class)
        else:
            return 0, 0


class TSNDataSet(data.Dataset):
    def __init__(self, data_path, list_file, num_dataload,
                 num_segments=3, total_segments=25, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.t7', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, noun_data_path=None):
        """

        """
        self.modality = modality
        try:
            with open(data_path, "rb") as f:
                # read dictionary having the data. It has features and the narration_id
                data: dict = pickle.load(f)
                if modality == "ALL":
                    self.data: np.ndarray = np.concatenate(list(data['features'].values()), -1)
                else:
                    self.data: np.ndarray = data['features'][modality]
                data_narrations: [] = data['narration_ids']
                data_narrations: [] = [narration.split("_")[-1] for narration in data_narrations]
                # Create a dictionary having the id of the narration and the corresponding data.
                self.data = dict(zip(data_narrations, self.data))
            if noun_data_path is not None:
                with open(noun_data_path, "rb") as f:
                    data = pickle.load(f)
                    if modality == "ALL":
                        self.noun_data = np.concatenate(list(data['features'].values()), -1)
                    else:
                        self.noun_data = data['features'][modality]
                    data_narrations = data['narration_ids']
                    self.noun_data = dict(zip(data_narrations, self.noun_data))
            else:
                self.noun_data = None
        except:
            raise Exception("Cannot read the data in the given pickle file {}".format(data_path))

        self.list_file = list_file
        self.num_segments = num_segments
        self.total_segments = total_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1  # Diff needs one more image to calculate diff
        # Read the labels
        self._parse_list()  # read all the video files

    # def _load_feature(self, directory, idx):
    #     if self.modality == 'RGB' or self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
    #         feat_path = os.path.join(directory, self.image_tmpl.format(idx))
    #         try:
    #             feat = [torch.load(feat_path)]
    #         except:
    #             print(Back.RED + feat_path)
    #         return feat
    #
    #     elif self.modality == 'Flow':
    #         x_feat = torch.load(os.path.join(directory, self.image_tmpl.format('x', idx)))
    #         y_feat = torch.load(os.path.join(directory, self.image_tmpl.format('y', idx)))
    #
    #         return [x_feat, y_feat]

    def load_features_noun(self, idx, segment):
        return torch.from_numpy(np.expand_dims(self.noun_data[idx][segment - 1], axis=0)).float()

    def _load_feature(self, idx, segment):
        return torch.from_numpy(np.expand_dims(self.data[idx][segment - 1], axis=0)).float()

    def _parse_list(self):
        try:
            # Read the file containing the label
            label_file: pd.DataFrame = pd.read_pickle(self.list_file).reset_index()
            self.labels_available = (("verb_class" in label_file))# and ("noun_class" in label_file)) fixme : Modification to not take nouns into consideration
        except:
            raise Exception("Cannot read pickle, {},containing labels".format(self.list_file))

        # Get the available records
        available_records = self._getAvailableFeatures()

        # Initialize the video records using the pandas series associated to a given row
        self.video_list = [VideoRecord(i, row[1], self.total_segments) for i, row in enumerate(label_file.iterrows())]
        # repeat the list if the length is less than num_dataload (especially for target data)
        n_repeat = self.num_dataload // len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list * n_repeat + self.video_list[:n_left]

    def _readOurPickle(self):
        try:
            labels_filed1: pd.DataFrame = pd.read_pickle("/home/andres/MLDL/EGO_Project_Group4/train_val/D1_train.pkl")
            labels_filed2: pd.DataFrame = pd.read_pickle("/home/andres/MLDL/EGO_Project_Group4/train_val/D2_train.pkl")
            labels_filed3: pd.DataFrame = pd.read_pickle("/home/andres/MLDL/EGO_Project_Group4/train_val/D3_train.pkl")
            # Concatenate tables
            frames = [labels_filed1, labels_filed2, labels_filed3]
            result = pd.concat(frames)
        except:
            raise Exception("Cannot read pickle, {},containing labels".format(self.list_file))

        return result

    def _getAvailableFeatures(self):
        """ Get the available videos

        """
        self.uids = {key.split("_")[-1]: value for key, value in self.data.items()}
        available_records: list = list(self.uids.keys())

        # Get the participant id

        return available_records

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        # np.random.seed(1)
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x)) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x)) for x in
                                range(self.num_segments)])  # pick the central frame in each segment
        else:  # the video clip is too short --> duplicate the last frame
            id_select = np.array([x for x in range(num_select)])
            # expand to the length of self.num_segments with the last element
            id_expand = np.ones(self.num_segments - num_select, dtype=int) * id_select[id_select[0] - 1]
            offsets = np.append(id_select, id_expand)

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        self.myGet(index=index,record=record)
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        frames = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_feats = self._load_feature(record.segment_id, p)
                frames.extend(seg_feats)

                if p < record.num_frames:
                    p += 1

        # process_data = self.transform(frames)
        process_data_verb = torch.stack(frames)

        frames = list()

        if self.noun_data is not None:
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_feats = self.load_features_noun(record.segment_id, p)
                    frames.extend(seg_feats)

                    if p < record.num_frames:
                        p += 1

            # process_data = self.transform(frames)
            process_data_noun = torch.stack(frames)
            process_data = [process_data_verb, process_data_noun]
        else:
            process_data = process_data_verb

        return process_data, record.label, record.segment_id

    def myGet(self,record,index:int):
        """
        Function for getting the processed data, the label and the index
        """
        if index < 0 or index > len(self.video_list):
            raise IndexError

        torch_tensor = torch.from_numpy(self.data[index])

    def __len__(self):
        return len(self.video_list)



class TSNDataSetModified(data.Dataset):
    def __init__(self, data_path, list_file, num_dataload,
                 num_segments=3, total_segments=25, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.t7', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, noun_data_path=None):
        """

        """
        self.modality = modality
        try:
            with open(data_path, 'rb') as f:
                self.raw_data = pickle.load(f)
        except IOError:
            print(f"File not found : {data_path}. Try to check the file exists and the path is correct.")
            import sys
            sys.exit()

        # Get data
        self.data: np.ndarray = self.raw_data['features'][modality]
        self.narration_ids = self.raw_data['narration_ids']

        labelsDF: pd.DataFrame = pd.read_pickle(list_file)

        ids = labelsDF["uid"]
        # todo : add check that uid correspond to the narration id
        self.labels = labelsDF["verb_class"]
        self.labels_available = True

        self.list_file = list_file
        self.num_segments = num_segments
        self.total_segments = total_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1  # Diff needs one more image to calculate diff


    def __getitem__(self, index):
        if index < 0 or index > len(self.labels):
            raise IndexError
        # Cast to torch tensor from numpy array to use in torch models.
        torch_tensor = torch.from_numpy(self.data[index])
        #todo : For efficiency it is better to do this transformations when initializing the dataset, change it

        # Transpose to have as the first dimension the clips
        #torch_tensor = torch.transpose(torch_tensor, 0, 1).float()

        label_vector = torch.tensor([self.labels[index]], dtype=torch.long)
        label_noun = torch.tensor([1],dtype=torch.long) # nothing

        return torch_tensor.float(), (label_vector.long(),label_noun.long()),1

    def __len__(self):
        return len(self.labels)