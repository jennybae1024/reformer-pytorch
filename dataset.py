import warnings
warnings.filterwarnings("ignore")
import json
import torch
from torch.utils.data import Dataset


class DatasetForLM(Dataset):
    def __init__(self, tokenizer, max_len):
        print('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []

        data = json.load(open('FloDial-dataset/dialogs/dialogs.json'))
        for key in data.keys():
            for utt in data[key]["utterences"]:
                self.docs.append(utt["utterance"])

        print('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, add_special_tokens:bool = False, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        labels = inputs.clone()

        inputs= inputs.squeeze()
        labels= labels.squeeze()
        inputs_mask = inputs != 0

        return inputs, labels, inputs_mask