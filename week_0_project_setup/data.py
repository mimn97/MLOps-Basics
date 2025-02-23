import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer

# Using the Lightning Data Module -> automate distribution of workload between CPU and GPU
class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola") # download the dataset
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example): # Transforms by tokenize
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        # accepts stage to define if we are at fit or test step
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self): # wrap inside a DataLoader
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data() # Download the dataset
    data_model.setup() # Apply transforms (tokenize)
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
