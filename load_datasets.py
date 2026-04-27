import json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, Value
from pathlib import Path

RANDOM_STATE = 42


def load_m4gt(data_dir: str = "M4GT-Bench", val_split: float = 0.2, seed: int = 42):
    """
    Load M4GT-Bench dataset.
    
    Paper: https://arxiv.org/pdf/2402.11175
    GitHub: https://github.com/mbzuai-nlp/M4GT-Bench
    
    Google Drive: https://drive.google.com/drive/folders/1hBgW6sgZfz1BK0lVdUu0bZ4HPKSpOMSY
    
    Available files:
        - SubtaskA_multilingual.jsonl (multilingual)
        - SubtaskA.jsonl (English only)
        - SubtaskB.jsonl
        - subtaskC_test.jsonl
        - subtaskC_train_dev.jsonl
    
    Labels: 1 - Machine-generated, 0 - Human-written
    
    Args:
        file_path: Path to the JSONL file (e.g., "M4GT-Bench/SubtaskA_multilingual.jsonl")
    
    Returns:
        HuggingFace Dataset
    """
    data_path = Path(data_dir)

    data_file = data_path / "SubtaskA_multilingual.jsonl"

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    texts = []
    labels = []

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(int(data["label"]))

    ds = Dataset.from_dict({"text": texts, "label": labels})

    ds = ds.shuffle(seed=seed)
    ds = ds.train_test_split(
        test_size=val_split,
        seed=seed,
    )

    return DatasetDict({
        "train": ds["train"],
        "validation": ds["test"]
    })


def load_ruatd():
    """
    Load RuATD dataset.

    Paper: https://arxiv.org/pdf/2206.01583
    Hugging Face: https://huggingface.co/datasets/RussianNLP/coat

    Labels: 1 - Machine-generated, 0 - Human-written

    Returns:
        DatasetDict with train, validation, test splits
    """
    ds = load_dataset("RussianNLP/coat", "binary")

    return DatasetDict({
        "train": ds["train"],
        "validation": ds["validation"],
        "test": ds["test"]
    })


def load_mgt_multi():
    """
    Load MGT-1 Multi.

    Hugging Face: https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_multingual

    Labels: 1 - Machine-generated, 0 - Human-written

    Returns:
        DatasetDict with train, validation splits
    """
    ds = load_dataset("Jinyan1/COLING_2025_MGT_multingual")

    return DatasetDict({
        "train": ds["train"],
        "validation": ds["dev"]
    })


def load_mage():
    """
    Load MAGE dataset.

    Paper: https://arxiv.org/pdf/2305.13242
    GitHub: https://github.com/yafuly/MAGE
    Hugging Face: https://huggingface.co/datasets/yaful/MAGE

    Original labels: 1 - Human-written, 0 - Machine-generated
    Converted labels: 1 - Machine-generated, 0 - Human-written (for consistency)

    Returns:
        DatasetDict with train, validation, test splits
    """
    ds = load_dataset("yaful/MAGE")

    # Convert labels: 0 - machine-generated, 1 - human-written
    # Replace to 1 - machine-generated, 0 - human-written for consistency
    ds = ds.map(lambda x: {"label": 1 - x["label"]})

    return DatasetDict({
        "train": ds["train"],
        "validation": ds["validation"],
        "test": ds["test"]
    })

def load_raid(val_split: float = 0.2, seed: int = 42):
    """
    Load RAID dataset.

    Paper: https://arxiv.org/abs/2405.07940
    Github: https://github.com/liamdugan/raid
    Hugging Face: https://huggingface.co/datasets/liamdugan/raid
    """
    ds = load_dataset("liamdugan/raid", split="train[:10%]")

    def add_label(example):
        example["label"] = 0 if example["model"] == "human" else 1
        return example
    
    ds = ds.map(add_label)
    ds = ds.rename_column("generation", "text")

    ds = ds.shuffle(seed=seed)
    ds = ds.train_test_split(
        test_size=val_split,
        seed=seed,
    )

    return DatasetDict({
        "train": ds["train"],
        "validation": ds["test"]
    })

def load_iberautextification():
    """
    Load IberAuTexTification dataset.

    Paper: https://riunet.upv.es/server/api/core/bitstreams/9d2f6cb3-ad7c-417f-83a7-fb4f55190032/content
    Github: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
    Hugging Face: https://huggingface.co/datasets/Hello-SimpleAI/HC3
    """

    ds_train = load_dataset(
        "csv",
        data_files="hf://datasets/Genaios/iberautextification/data/subtask_1/train.tsv",
        sep="\t"
    )["train"]
    ds_test = load_dataset(
        "csv",
        data_files="hf://datasets/Genaios/iberautextification/data/subtask_1/test.tsv",
        sep="\t"
    )["train"]

    def replace_label(example):
        example["label"] = 0 if example["label"] == "human" else 1
        return example
    
    ds_train = ds_train.map(replace_label)
    ds_test= ds_test.map(replace_label)


    return DatasetDict({
        "train": ds_train,
        "validation": ds_test,
    })

def load_h3():
    """
    Load H3 dataset.

    Paper: https://arxiv.org/abs/2301.07597
    Github: https://github.com/Genaios/IberAuTexTification/tree/main
    Hugging Face: https://huggingface.co/datasets/Genaios/iberautextification
    """
    pass

def sample_balanced_dataset(ds: Dataset, sample_size: int = 1000, seed: int = RANDOM_STATE):
    ds_class_0 = ds.filter(lambda x: x["label"] == 0)
    ds_class_1 = ds.filter(lambda x: x["label"] == 1)

    ds_sample_0 = ds_class_0.shuffle(seed=seed).select(range(sample_size//2))
    ds_sample_1 = ds_class_1.shuffle(seed=seed).select(range(sample_size//2))

    ds_balanced = concatenate_datasets([ds_sample_0, ds_sample_1])
    ds_balanced = ds_balanced.shuffle(seed=seed)

    return ds_balanced


def clean_dataset(dataset, sample_size, seed):
    keep_cols = ["text", "label"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])
    features = Features({
        "text": Value("string"),
        "label": Value("int64"),
    })
    dataset = dataset.cast(features)
    sampled = sample_balanced_dataset(dataset, sample_size, seed)
    return sampled


def create_balanced_dataset(sample_size: int = 1000, seed: int = RANDOM_STATE):
    datasets_dict = {
        "m4gt": load_m4gt(),
        "ruatd": load_ruatd(),
        "mgt_multi": load_mgt_multi(),
        "mage": load_mage(),
        "raid": load_raid(),
        "iberautextification": load_iberautextification(),
    }

    train_datasets = []
    val_datasets = []

    for ds in datasets_dict.values():
        train_datasets.append(clean_dataset(ds["train"], sample_size, seed))
        val_datasets.append(clean_dataset(ds["validation"], sample_size, seed))

    ds_combined_train = concatenate_datasets(train_datasets).shuffle(seed=seed)
    ds_combined_val = concatenate_datasets(val_datasets).shuffle(seed=seed)

    return ds_combined_train, ds_combined_val