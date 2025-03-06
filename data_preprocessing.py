import tiktoken
import torch


def load_lyrics(file_path):
    """Load and read the lyrics from a text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        lyrics = f.read()
    return lyrics


def encode_data(data, tokeniser):
    """Encode the data (lyrics) using the tokeniser provided"""
    return tokeniser.encode(data)


def decode_data(data, tokeniser):
    return tokeniser.decode(data)


def get_batch(data, batch_size, block_size, split="train"):
    """generate a small batch of data for inputs x and targets y"""

    def split_data(data, split="train"):
        """If split is not train then assume its validation split"""
        n = int(0.9 * len(data))
        if split == "train":
            return data[:n]
        return data[n:]

    data = split_data(data, split)

    # generate a tensor of random integers, that represent the starting position of each sequence of data
    batch_start_indicies = torch.randint(
        high=(len(data) - block_size), size=(batch_size,)
    )

    # creates a tensor: x and y where each element is a sequence of block_size, stack the 1-D tensors as rows
    # creating a batch_size x block_size tensors (e.g 4x8 tensor)
    x = torch.stack(
        [data[index : index + block_size] for index in batch_start_indicies]
    )
    y = torch.stack(
        [data[index + 1 : index + block_size + 1] for index in batch_start_indicies]
    )
    return x, y
