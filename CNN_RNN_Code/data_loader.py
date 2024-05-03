# Imports
import torch
import spacy
import os
from PIL import Image
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# For tokenizer
# After installing spacy, run this command on terminal to load this model
# python -m spacy download en_core_web_sm
spacy_english = spacy.load("en_core_web_sm")

# Constants
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
IMAGE_FOLDER = "../input/flickr8k/Images"


# class to build the vocabulary from the captions and perform encoding
class VocabBuilder:
    def __init__(self, freq_threshold=5):
        # For encoding and decoding
        self.idx_word_map = {0: START_TOKEN, 1: END_TOKEN, 2: UNK_TOKEN, 3: PAD_TOKEN}
        self.word_idx_map = {START_TOKEN: 0, END_TOKEN: 1, UNK_TOKEN: 2, PAD_TOKEN: 3}

        # threshold to keep track of frequent words
        self.threshold = freq_threshold

    def __len__(self):
        return len(self.idx_word_map)

    def create_vocab(self, caption_list):
        # After special constants this will be our index
        idx = 4
        frequency_map = defaultdict(int)
        # Encode information based on threshold, if higher than threshold add to the maps
        for caption in caption_list:
            caption_tokenized = [
                token.text.lower() for token in spacy_english.tokenizer(caption)
            ]
            for word in caption_tokenized:
                frequency_map[word] += 1
                if frequency_map[word] == self.threshold:
                    self.idx_word_map[idx] = word
                    self.word_idx_map[word] = idx
                    idx += 1

    # Perform encoding and if lower than threshold encode with <UNK> token
    def encode_text(self, caption):
        caption_tokenized = [
            token.text.lower() for token in spacy_english.tokenizer(caption)
        ]
        encoded_result = []
        for token in caption_tokenized:
            if token in self.word_idx_map:
                encoded_result.append(self.word_idx_map[token])
            else:
                encoded_result.append(self.word_idx_map[UNK_TOKEN])
        return encoded_result


# Dataset transformation and retreival
class FlickrDataset(Dataset):
    def __init__(self, dataframe, transform, freq_threshold=5):
        self.transform = transform
        self.dataframe = dataframe
        self.flickr_transform = transform
        self.images = self.dataframe["image"]
        self.captions = self.dataframe["caption"]
        # Build the vocabulary
        self.vocabulary = VocabBuilder(freq_threshold)
        self.vocabulary.create_vocab(self.captions)

    def __len__(self):
        return len(self.dataframe)

    # For each item retrieval, transform the image and encode the caption correctly
    def __getitem__(self, index):
        caption = self.captions[index]
        image_name = self.images[index]
        image = Image.open(os.path.join(IMAGE_FOLDER, image_name)).convert("RGB")
        image = self.transform(image)

        # Add start and end tokens to each caption
        encoded_caption = [self.vocabulary.word_idx_map[START_TOKEN]]
        encoded_caption += self.vocabulary.encode_text(caption)
        encoded_caption.append(self.vocabulary.word_idx_map[END_TOKEN])

        return image, torch.tensor(encoded_caption)


# Collate function which performs preprocessing on batches and pads the captions to have same length
class CustomCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        # Reduce dimension from batch to get the images
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        # Pad sequence to have all captions the same length which should be equal to max length in each batch
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_index)

        return images, targets


# Data loader function which takes the dataset, batch_size, tranform function and the worker_number
def get_data_loader(dataset, transform, batch_size=32, num_workers=4):
    curr_dataset = FlickrDataset(dataset, transform)
    pad_index = curr_dataset.vocabulary.word_idx_map[PAD_TOKEN]
    # Initiate dataloader and return the data_loader and the dataset
    data_loader = DataLoader(
        dataset=curr_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=CustomCollate(pad_index),
    )
    return data_loader, curr_dataset
