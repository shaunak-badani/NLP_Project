from make_dataset import DatasetProcessor
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer, \
         GPT2LMHeadModel, TrainingArguments, Trainer
import torch

class ModelTrainer:
    
   @staticmethod
   def load_data_collator(tokenizer, mlm = False):
      data_collator = DataCollatorForLanguageModeling(
         tokenizer=tokenizer, 
         mlm=mlm,
      )
      return data_collator
   
   @staticmethod
   def get_parameters_to_train(model):
      trainable_weight_names = set(['transformer.wte.weight'])
      trainable_weights = [param for name, param in model.named_parameters() if name in trainable_weight_names]
      return trainable_weights
   
   @staticmethod
   def train(train_file_path,
            model_name,
            per_device_train_batch_size,
          num_train_epochs):
      tokenizer = GPT2Tokenizer.from_pretrained(model_name)
      train_dataset = DatasetProcessor.load_dataset(train_file_path, tokenizer)
      data_collator = ModelTrainer.load_data_collator(tokenizer) 
      model = GPT2LMHeadModel.from_pretrained(model_name)
      training_args = TrainingArguments(
         output_dir = "/tmp",
         save_strategy = "no",
         per_device_train_batch_size=per_device_train_batch_size,
         num_train_epochs=num_train_epochs
      )
      optimizer = AdamW(
         ModelTrainer.get_parameters_to_train(model), 
         lr = 1e-5)
      trainer = Trainer(
               model=model,
               args=training_args,
               data_collator=data_collator,
               train_dataset=train_dataset,
               optimizers = (optimizer, None)
      ) 
      trainer.train()
      torch.save(model, "./models/fine_tuned_model.pt")
   

if __name__ == "__main__":
   model_name = 'gpt2'
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)
   input_file_path = "./data/Articles.csv"
   train_file_path = "./data/Articles.txt"
   DatasetProcessor.process_data(input_file_path, train_file_path)
   train_dataset = DatasetProcessor.load_dataset(train_file_path, tokenizer)

   per_device_train_batch_size = 8
   num_train_epochs = 1.0
   ModelTrainer.train(train_file_path, model_name, per_device_train_batch_size, num_train_epochs)

