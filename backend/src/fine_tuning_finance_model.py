## This code is for demonstration purposes. The actual model was fine-tuned with Google Colab T4-GPU and saved in Hugging Face Hub.
import os
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from datasets import load_dataset, concatenate_datasets

def main():
    # Load API key from environment variable
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the HUGGINGFACE_API_KEY environment variable.")

    # Load dataset from the hub
    dataset = load_dataset("philschmid/finanical-rag-embedding-dataset", split="train")
    dataset = dataset.rename_column("question", "anchor")
    dataset = dataset.rename_column("context", "positive")
    dataset = dataset.add_column("id", range(len(dataset)))
    dataset = dataset.train_test_split(test_size=0.1)
    dataset["train"].to_json("train_dataset.json", orient="records")
    dataset["test"].to_json("test_dataset.json", orient="records")

    # Load model
    model_id = "BAAI/bge-base-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)

    # Inform user about the device being used
    if device == "cuda":
        print("Using GPU for training.")
    else:
        print("Using CPU for training.")

    # Load datasets
    test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
    train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Prepare queries and corpus
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
    relevant_docs = {q_id: [q_id] for q_id in queries}

    # Define evaluators
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
        )
        matryoshka_evaluators.append(ir_evaluator)
    evaluator = SequentialEvaluator(matryoshka_evaluators)

    # Evaluate the model
    results = evaluator(model)
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]}")

    # Define loss function
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

    # Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="bge-base-financial-matryoshka",
        num_train_epochs=4,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
    )

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),
        loss=train_loss,
        evaluator=evaluator,
    )

    # Start training
    trainer.train()
    trainer.save_model()
    trainer.model.push_to_hub("bge-base-financial-matryoshka")

if __name__ == "__main__":
    main()
