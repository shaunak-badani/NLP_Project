import pandas as pd
import numpy as np
import re
from transformers import TextDataset

class DatasetProcessor:
    """
    Class for processing and cleaning dataset
    """

    @staticmethod
    def cleaning(s):
        """Function to clean the string passed to it"""
        s = str(s)
        s = re.sub('\s\W',' ',s)
        s = re.sub('\W,\s',' ',s)
        s = re.sub("\d+", "", s)
        s = re.sub('\s+',' ',s)
        s = re.sub('[!@#$_]', '', s)
        s = s.replace("co","")
        s = s.replace("https","")
        s = s.replace("[\w*"," ")
        return s

    @staticmethod
    def process_data(file_path, output_path):
        """Processes data from csv and outputs it to a txt file"""
        df = pd.read_csv(file_path, encoding="ISO-8859-1") 
        df = df.dropna()

        text_data = open(output_path, 'w')
        for idx, item in df.iterrows():
            article = DatasetProcessor.cleaning(item["Article"])
            text_data.write(article)
        text_data.close()

    @staticmethod
    def load_dataset(file_path, tokenizer, block_size = 128):
        dataset = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset

if __name__ == "__main__":
    file_path = "./data/Articles.csv"
    output_path = "./data/Articles_processed.txt"
    print(f"Processing data from {file_path} and storing it in {output_path}")
    DatasetProcessor.process_data(file_path, output_path)
