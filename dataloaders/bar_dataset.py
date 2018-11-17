import music21
import torch

from music21 import note
from torch.utils.data import TensorDataset, DataLoader
from fractions import Fraction

# define global constants
MAX_NOTES = 1000
SLUR_SYMBOL = '__'
TIME_SIG_NUM = 3
TIME_SIG_DEN = 4
tick_values = [
    0,
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(3, 4)
]


class BarDataset():
    def __init__(self):
        # define constants
        self.time_sig_num = TIME_SIG_NUM
        self.time_sig_den = TIME_SIG_DEN
        
        # instantiate tick durations
        self.beat_subdivisions = len(tick_values)
        self.tick_durations = self.compute_tick_durations()

        # load the note and index dictionaries
        self.dict_path = 'dat/dicts.txt'
        f = open(self.dict_path, 'r')
        dicts = [line.rstrip('\n') for line in f]
        self.index2note_dicts = eval(dicts[0])
        self.note2index_dicts = eval(dicts[1])
        
        # load the dataset
        self.dataset_path = self.__repr__()
        self.dataset = torch.load(self.dataset_path)

    def __repr__(self):
        return 'dat/BarDataset'    
    
    def get_dataset(self):
        """
        Returns the dataset 
        """
        return self.dataset()

    def empty_bar_tensor(self):
        """
        Creates an empty tensor for a bar

        :return: torch long tensor, initialized with start indices 
        """
        slur_symbols = self.note2index_dicts[SLUR_SYMBOL]
        slur_symbols = torch.from_numpy(slur_symbols).long().clone()
        bar_length = len(tick_values) * TIME_SIG_NUM
        slur_symbols = slur_symbols.repeat(bar_length, 1).transpose(0, 1)
        return slur_symbols

    def get_score_from_tensor(self, tensor_score):
        """
        Converts the given score tensor to a music21 score object
        :param tensor_score: torch tensor, (1, num_ticks)
        :return: music21 score object
        """
        slur_index = self.note2index_dicts[SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_score.numpy().flatten()
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.beat_subdivisions]
                f = BarDataset.standard_note(self.index2note_dicts[note_index])
            else:
                dur += self.tick_durations[tick_index % self.beat_subdivisions]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    def data_loaders(self, batch_size, split=(0.7, 0.2)):
        """
        Returns three data loaders obtained by splitting
        self.dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1
        dataset = self.dataset
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                             int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl

    @staticmethod
    def standard_name(note_or_rest):
        """
        Convert music21 note objects to str
        
        :param note_or_rest: music21 note object,
        :return: str,
        """
        if isinstance(note_or_rest, music21.note.Note):
            return note_or_rest.nameWithOctave
        if isinstance(note_or_rest, music21.note.Rest):
            return note_or_rest.name
    
    @staticmethod
    def standard_note(note_or_rest_string):
        """
        Convert note or rest string to music21 object
        :param note_or_rest_string: str,
        :return: music21 note object,
        """
        if note_or_rest_string == 'rest':
            return note.Rest()
        elif note_or_rest_string == SLUR_SYMBOL:
            return note.Rest()
        else:
            return note.Note(note_or_rest_string)

    @staticmethod
    def compute_tick_durations():
        """
        Computes the tick durations
        """
        diff = [n - p
                for n, p in zip(tick_values[1:], tick_values[:-1])]
        diff = diff + [1 - tick_values[-1]]
        return diff

