import os
import sys
import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch
import datasets
import huggingface_hub
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# qwen-vl utilities (shipped with the notebook environment)
from qwen_vl_utils import process_vision_info


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (verbose)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    root.addHandler(fh)

    logging.debug("Logging initialized. Log file at %s", log_path)


def login_huggingface(token_env_var: str = "HUGGINGFACE_TOKEN") -> None:
    token = os.environ.get(token_env_var)
    if not token:
        logging.warning("Environment variable %s not set; proceeding without HF login.", token_env_var)
        return
    try:
        huggingface_hub.login(token)
        logging.info("Logged into Hugging Face Hub.")
    except Exception as e:
        logging.exception("Failed to login to Hugging Face Hub: %s", e)


def load_dataset() -> datasets.Dataset:
    logging.info("Loading dataset kahua-ml/flattened_nameplate_dataset (split=train)…")
    dataset = datasets.load_dataset("kahua-ml/flattened_nameplate_dataset", split="train")
    logging.info("Dataset loaded: %d examples", len(dataset))
    return dataset


def load_model_and_processor():
    logging.info("Loading Qwen2.5-VL-7B-Instruct model and processor…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    logging.info("Model and processor loaded.")
    return model, processor


def build_peft_model(model):
    logging.info("Configuring LoRA (PEFT)…")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.enable_input_require_grads()
    peft_model = get_peft_model(model, peft_config)
    logging.info("PEFT model prepared.")
    return peft_model


def text_formatter(field: str) -> str:
    return (
        f"You are a helpful assistant that extracts text from images and returns JSON for programmatic use. "
        f"Find the value of {field} and return a JSON string in the format of {{'{field}': '<value>'}}."
    )


def answer_formatter(field: str, value: str) -> str:
    return f"{{'{field}': '{value}'}}"


class MyDataCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor

    def __call__(self, examples) -> Any:
        texts: List[str] = []
        images: List[Any] = []
        assistant_responses: List[str] = []

        for example in examples:
            image = example["image"]
            field = example["field"]
            value = example["value"]
            answer = answer_formatter(field, value)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_formatter(field)},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            images += process_vision_info(messages)[0]
            assistant_responses.append(answer)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()

        for i, (input_ids, assistant_response) in enumerate(zip(batch["input_ids"], assistant_responses)):
            assistant_tokens = self.processor.tokenizer(assistant_response, return_tensors="pt")["input_ids"][0]
            start_idx = self.find_subsequence(input_ids, assistant_tokens)
            if start_idx is not None:
                labels[i, : start_idx] = -100
                labels[i, start_idx + len(assistant_tokens) :] = -100

        batch["labels"] = labels
        return batch

    def find_subsequence(self, sequence: torch.Tensor, subsequence: torch.Tensor) -> Optional[int]:
        seq_len = len(sequence)
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i : i + sub_len], subsequence):
                return i
        return None


class SaveBestCallback(TrainerCallback):
    def __init__(self, output_dir: str, processor: AutoProcessor, peft_model, metric_name: str = "eval_loss") -> None:
        self.metric_name = metric_name
        self.best_metric = float("inf")
        self.output_dir = output_dir
        self.best_dir = os.path.join(output_dir, "best")
        os.makedirs(self.best_dir, exist_ok=True)
        self.processor = processor
        self.peft_model = peft_model

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            model = kwargs.get("model", self.peft_model)
            logging.info("New best %s=%.6f. Saving best artifacts to %s", self.metric_name, metric_value, self.best_dir)
            try:
                model.save_pretrained(self.best_dir)
            except Exception as e:
                logging.warning("Failed to save best model: %s", e)
            try:
                self.peft_model.save_pretrained(os.path.join(self.best_dir, "adapter"))
            except Exception as e:
                logging.warning("Failed to save best adapter: %s", e)
            try:
                self.processor.save_pretrained(self.best_dir)
            except Exception as e:
                logging.warning("Failed to save best processor: %s", e)
        return control


def main():
    setup_logging(log_dir=os.path.join(os.getcwd(), "logs"))

    login_huggingface()

    dataset = load_dataset()

    logging.info("Creating train/test split (80/20)…")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_dataset["train"].shuffle()
    test_ds = split_dataset["test"].shuffle()
    logging.info("Train examples: %d | Eval examples: %d", len(train_ds), len(test_ds))

    model, processor = load_model_and_processor()
    peft_model = build_peft_model(model)

    data_collator = MyDataCollator(processor)

    output_dir = os.path.join(os.getcwd(), "qwen-peft-vl")

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps",
        save_steps=80,
        save_total_limit=2,
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        output_dir=output_dir,
        report_to=["none"],
        logging_first_step=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_ds,
        train_dataset=train_ds,
        callbacks=[SaveBestCallback(output_dir=output_dir, processor=processor, peft_model=peft_model, metric_name="eval_loss")],
    )

    logging.info("Starting training…")
    train_result = trainer.train()
    logging.info("Training completed. %s", str(train_result))

    logging.info("Saving model, adapter, and processor…")
    # Save trainer model (PEFT weights are in the wrapped model)
    trainer.save_model(output_dir)

    # Save only the PEFT adapter weights separately for clarity
    try:
        peft_model.save_pretrained(os.path.join(output_dir, "adapter"))
    except Exception as e:
        logging.warning("Could not save separate adapter folder: %s", e)

    # Save processor/tokenizer
    processor.save_pretrained(output_dir)
    logging.info("Artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()


