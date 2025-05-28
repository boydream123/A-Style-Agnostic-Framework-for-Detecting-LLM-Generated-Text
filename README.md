# A-Style-Agnostic-Framework-for-Detecting-LLM-Generated-Text

# Content

- [Universal Dataset Builder for Human vs. LLM Text](#universal-dataset-builder-for-human-vs-llm-text)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [1. Building a Dataset](#1-building-a-dataset)
    - [2. Analyzing an Existing Dataset](#2-analyzing-an-existing-dataset)
    - [3. Splitting a Dataset](#3-splitting-a-dataset)
  - [Output Format (JSONL)](#output-format-jsonl)
  - [Dataset Dictionary Structure](#dataset-dictionary-structure)

- [SAFD](#safd)
  - [Overview](#overview)
  - [Features](#features-1)
  - [System Requirements](#system-requirements)
  - [Configuration](#configuration-1)
  - [Data Format](#data-format)
  - [Model Architecture](#model-architecture)
  - [Loss Functions](#loss-functions)
  - [Utility Functions](#utility-functions)
  - [Training and Evaluation](#training-and-evaluation)
  - [Main Execution](#main-execution)
  - [Output](#output)
  - [LoRA Fine-Tuning](#lora-fine-tuning)

# Universal Dataset Builder for Human vs. LLM Text

This project provides a Python-based toolkit for building datasets composed of human-written text and corresponding text rewritten by Large Language Models (LLMs). It fetches data from sources like Project Gutenberg and X-Sum, processes it, and uses a configurable LLM API to generate stylized text variations. The primary goal is to create datasets useful for tasks such as detecting AI-generated text or analyzing stylistic differences.

## Features

* **Data Ingestion:** Loads text samples from Hugging Face datasets (e.g., Gutenberg, X-Sum).
* **Configurable LLM API Integration:**
    * Supports multiple API formats (OpenAI, Anthropic, and custom).
    * Manages API keys, endpoints, and request parameters via a JSON configuration file.
    * Handles API call retries with exponential backoff and rate limiting.
* **Text Rewriting:** Uses a customizable prompt to instruct an LLM to rewrite text in a specific style (default: Hans Christian Andersen fairy tale style with defined characteristics).
* **Dataset Creation:**
    * Generates pairs of original human-written text and LLM-rewritten text.
    * Labels samples as human-written (0) or LLM-generated (1).
    * Saves the final dataset in JSONL format.
* **Utilities:**
    * Dataset statistics (total samples, counts by label/source, text length analysis).
    * Functions to analyze existing JSONL datasets.
    * Function to split datasets into training, validation, and test sets.
* **Logging:** Comprehensive logging for monitoring the dataset creation process.

## Prerequisites

* Python 3.7+
* pip (Python package installer)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/boydream123/A-Style-Agnostic-Framework-for-Detecting-LLM-Generated-Text.git
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The script uses an `api_config.json` file to manage LLM API settings.

1.  **Copy the example configuration file:**
    ```bash
    cp api_config.json.example api_config.json
    ```

2.  **Edit `api_config.json`** with your specific API details:
    ```json
    {
        "api_url": "YOUR_API_ENDPOINT_URL", // e.g., "[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)"
        "model_name": "YOUR_MODEL_NAME",    // e.g., "gpt-3.5-turbo" or "claude-2"
        "api_key": "YOUR_API_KEY",          // Your secret API key
        "headers": {
            "Content-Type": "application/json"
            // Add other necessary headers here, e.g., "Authorization" if not handled by request_format
        },
        "request_format": "openai",  // Supported: "openai", "anthropic", "custom"
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 60,               // Timeout for API requests in seconds
        "max_retries": 3,            // Number of retries for failed API calls
        "retry_delay": 5,            // Initial delay between retries in seconds
        "rate_limit_delay": 1,       // Delay between consecutive API calls in seconds
        "custom_prompt": "",         // Optional: Override the default rewriting prompt. Use {text} as placeholder.
        "request_template": {},      // For "custom" request_format: Define the JSON body structure. Use {model}, {prompt}, {temperature}, {max_tokens} as placeholders.
        "response_path": []          // For "custom" request_format: JSON path to extract the LLM's response text, e.g., ["choices", 0, "message", "content"]
    }
    ```
    * **`api_url`**: The endpoint URL for the LLM API.
    * **`model_name`**: The specific model you want to use.
    * **`api_key`**: Your API key for authentication. **Keep this secret!**
    * **`headers`**: Default headers for the API request. The script automatically adds `Authorization` for OpenAI or `x-api-key` for Anthropic if `api_key` is provided.
    * **`request_format`**: Specifies the API structure.
        * `openai`: For OpenAI-compatible APIs.
        * `anthropic`: For Anthropic Claude APIs.
        * `custom`: Allows defining a custom request body (`request_template`) and response extraction path (`response_path`).
    * **`custom_prompt`**: If you want to use a different system prompt for text rewriting, define it here. Ensure you include `{text}` as a placeholder for the original text.
    * **`request_template`**: (Only for `request_format: "custom"`) A JSON object defining the structure of the request payload. Use placeholders like `{model}`, `{prompt}`, `{temperature}`, `{max_tokens}` which will be filled by the script.
    * **`response_path`**: (Only for `request_format: "custom"`) A list of keys/indices to navigate the JSON response and extract the generated text.

    **Note:** The `data/` directory will be created by the script if it doesn't exist, to store downloaded Hugging Face datasets and output files. It's recommended to add `api_config.json` (the actual file with your key) and the `data/` directory to your `.gitignore` file to avoid committing sensitive information or large data files.

## Usage

The main script `universal_dataset_builder.py` can be run directly.

### 1. Building a Dataset

This is the primary function of the script. It loads human-written text, generates LLM versions, and saves the combined dataset.

```bash
python scripts/dataset_builder.py
```

You can modify the number of samples and output file name within the main() function in scripts/dataset_builder.py:
```python
# In main() function:
config = {
    'gutenberg_samples': 500,  # Number of Gutenberg samples
    'xsum_samples': 500,       # Number of X-Sum samples
    'output_file': 'human_vs_llm_dataset.jsonl'
}
dataset = builder.build_dataset(**config)
```

## 2. Analyzing an Existing Dataset
If you have an existing dataset in JSONL format, you can analyze it using the analyze_dataset function. Uncomment and adapt the example in the if __name__ == "__main__": block:
```python
# Example: Analyze an existing dataset
# analyzed_samples = analyze_dataset("path/to/your/dataset.jsonl")
```

### 3. Splitting a Dataset
To split an existing JSONL dataset into training, validation, and test sets, use the split_dataset function. Uncomment and adapt the example:
```python
# Example: Split a dataset
# split_dataset("human_vs_llm_dataset.jsonl", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
```


## Output Format (JSONL)
The generated dataset is saved in JSONL (JSON Lines) format. Each line in the file is a valid JSON object representing a single sample.

Sample structure:
```json
{
    "text": "The rewritten or original text content.",
    "source": "source_dataset_name" or "source_dataset_name_llm", // e.g., "gutenberg", "gutenberg_llm"
    "label": 0, // 0 for human-written, 1 for LLM-generated
    "original_text": "The original human text (only present if label is 1 and sample is LLM-generated)"
}
```

## Dataset Dictionary Structure

Each sample in the dataset (i.e., each line in the output `*.jsonl` file) is a JSON object with the following fields:

| Key             | Data Type | Description                                                                                                | Example                                     |
|-----------------|-----------|------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| `text`          | String    | The textual content of the sample. This can be either an original human-written text or an LLM-generated text. | "Once upon a time, in a land far away..." |
| `source`        | String    | Indicates the origin of the text. If it's an LLM-generated sample, `_llm` is appended to the original source name. | `"gutenberg"`, `"xsum"`, `"gutenberg_llm"`  |
| `label`         | Integer   | A binary label indicating whether the text is human-written or LLM-generated. `0` for human-written, `1` for LLM-generated. | `0`, `1`                                    |
| `original_text` | String    | (Optional) This field is present **only** if the sample is LLM-generated (`label: 1`). It contains the original human-written text that was provided to the LLM for rewriting. | "This is the original human sentence."      |

### Example JSONL entries:

**Human-written sample:**
```json
{"text": "The quick brown fox jumps over the lazy dog.", "source": "gutenberg", "label": 0}
```
**LLM-generated sample:**
```json
{"text": "In a realm of ancient lore, a swift, russet-furred creature did leap with graceful ease above a slumbering canine of indolent nature.", "source": "gutenberg_llm", "label": 1, "original_text": "The quick brown fox jumps over the lazy dog."}
```

```txt
=== Sample Preview ===

Sample 1:
Source: xsum
Label: 0 (Human-written)
Text length: 258
Text preview: 28 July 2017 Last updated at 08:36 BST
Luke, from Texas, lost his bear after his family's return flight from Colorado to Dallas.
But fortunately for Luke, his teddy bear was actually just on a little ...

Sample 2:
Source: xsum_llm
Label: 1 (LLM-generated)
Text length: 1741
Text preview: **The Tale of the Nightwatchmen and the Wayward Owl**  

Once upon a time, in the quiet town of Bracknell, where the streets lay still beneath the silver moon, a most peculiar incident occurred. A ban...
Original text preview: A social media post shows officers on night shift dealing with the bird after it was found in the middle of a road in Bracknell.
"Officers had a hoot last night removing this offender from obstructing...

Sample 3:
Source: xsum
Label: 0 (Human-written)
Text length: 499
Text preview: Kidd, who spent three months on loan at Palmerston Park last season, has signed a one-year contract, with Queens saying he had "interest elsewhere".
The defender told the Dumfries club's website: "I e...
```



# SAFD

SAFD is a Python script designed to distinguish between human-written and Large Language Model (LLM)-generated text. Beyond binary classification, it incorporates a fine-grained attribution mechanism to identify specific characteristics of LLM-generated content and a novel style alignment loss to improve robustness against human-authored texts that mimic LLM styles. The model leverages transformer architectures (specifically pre-trained models like Qwen2-7B) and supports Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation).


## Overview

The script implements a multi-task learning framework:

1.  **Binary Classification:** Classifies text as either human-written (label 0) or LLM-generated (label 1).
2.  **LLM Attribution:** For texts identified as LLM-generated, it predicts a set of pre-defined attribution labels (e.g., "Too structured logical structure", "Repetitive or patterned expression") indicating *why* it might be LLM-generated.
3.  **Style Alignment:** Introduces a loss term to encourage the model to produce similar classification outputs for an original human text and its stylistic variants. This helps the model learn core human style rather than superficial stylistic cues that might be easily mimicked or altered.

It uses a pre-trained transformer model as its backbone, with custom classification and attribution heads. LoRA can be optionally applied for efficient fine-tuning.

## Features

* **Binary Classification:** Human vs. LLM text detection.
* **Multi-Label Attribution:** Identifies characteristics of LLM-generated text based on predefined categories.
* **Style Alignment Loss:** Improves robustness by aligning predictions for original human texts and their stylistic variations.
* **LoRA Support:** Enables efficient fine-tuning of large models.
* **Transformer-Based:** Utilizes powerful pre-trained models from Hugging Face.
* **Comprehensive Evaluation:** Calculates accuracy, precision, recall, F1-score, and AUROC.
* **Confusion Matrix Plotting:** Visualizes classification performance.
* **Dummy Data Generation:** Includes a utility to create sample data for quick testing and demonstration.
* **Configurable:** Key parameters like model name, learning rate, epochs, LoRA settings, and loss weights are easily configurable.

## System Requirements

* Python 3.8+
* PyTorch (torch)
* Transformers (transformers)
* PEFT (peft)
* Scikit-learn (sklearn)
* NumPy (numpy)
* Matplotlib (matplotlib)




## Configuration

All major configurations are located at the beginning of the `SAFD.py` script.

* `MODEL_NAME`: Hugging Face model identifier (e.g., "Qwen/Qwen2-7B-Instruct") or path to a local model.
* `NUM_CLASSES`: Number of classes for the main classification task (default: 2 for Human/LLM).
* `NUM_ATTRIBUTION_LABELS`: Number of distinct attribution categories for LLM text (default: 5).
* `USE_LORA`: Boolean, set to `True` to enable LoRA fine-tuning.
* `LORA_R`: LoRA rank.
* `LORA_ALPHA`: LoRA alpha scaling factor.
* `LORA_DROPOUT`: Dropout probability for LoRA layers.
* `LORA_TARGET_MODULES`: List of module names in the base model to apply LoRA to.
* `LEARNING_RATE`: Optimizer learning rate.
* `EPOCHS`: Number of training epochs.
* `BATCH_SIZE`: Batch size for training and evaluation.
* `MAX_LENGTH`: Maximum sequence length for tokenization.
* `TEST_SPLIT_RATIO`: Proportion of data to use for the test set.
* `RANDOM_STATE_SPLIT`: Random seed for data splitting to ensure reproducibility.
* `ALPHA_STYLE`, `ALPHA_CLASS`, `ALPHA_ATTR`: Weights for the style alignment, classification, and attribution losses respectively in the combined loss function.
* `ATTRIBUTION_CATEGORIES`: List of strings describing each attribution label.
* `MODEL_SAVE_PATH`: Directory to save the trained model components.
* `METRICS_SAVE_PATH`: File path to save evaluation metrics as JSON.
* `PLOTS_SAVE_PATH`: Directory to save generated plots (e.g., confusion matrix).
* `DUMMY_DATA_PATH`: Path for the auto-generated dummy data file.
* `TRAIN_SPLIT_PATH`, `TEST_SPLIT_PATH`: Paths for saving the train and test data splits.

## Data Format

The script expects input data in **JSON Lines (`.jsonl`)** format. Each line in the file should be a valid JSON object representing a single data sample.

**Required fields for each sample:**

* `id` (string): A unique identifier for the text sample.
* `text` (string): The text content.
* `label` (integer): `0` for human-written, `1` for LLM-generated.

**Optional fields:**

* `attribution_labels` (list of integers): Required if `label` is `1` (LLM-generated) and you want to train the attribution task. This should be a list of binary values (0 or 1) of length `NUM_ATTRIBUTION_LABELS`, where each element corresponds to an attribution category.
* `is_styled_human_variant` (boolean): Set to `True` if this sample is a stylistic variation of an original human text. Default is `False`.
* `original_human_id` (string): Required if `is_styled_human_variant` is `True`. This should be the `id` of the original human text to which this variant is related. This is used for the style alignment loss.

**Example `dummy_data.jsonl` entries:**

```json
{"id": "h0", "text": "An original tale of a brave mouse 0.", "label": 0, "is_styled_human_variant": false}
{"id": "h0_sA", "text": "Once upon a time, mouse 0 ventured forth, with whiskers twitching.", "label": 0, "is_styled_human_variant": true, "original_human_id": "h0"}
{"id": "l0", "text": "The LLM analyzed data point 0, concluding statistical significance.", "label": 1, "attribution_labels": [1, 1, 0, 0, 0]}
```

`utils.data.Dataset`.
* Loads data from a list of dictionaries (parsed from `.jsonl`).
* Tokenizes text using the specified Hugging Face tokenizer.
* Pads/truncates sequences to `MAX_LENGTH`.
* Formats data into tensors for `input_ids`, `attention_mask`, `class_label`, and `attribution_labels`.
* Identifies groups of styled human variants based on `original_human_id` for the style alignment loss.
* `from_jsonl` class method to conveniently load data from a `.jsonl` file.

**`safd_collate_fn(batch)`**:
* Custom collate function for the DataLoader.
* Stacks batched tensors for `input_ids`, `attention_mask`, etc.
* Creates masks (`has_attribution_mask`, `is_llm_generated_mask`).
* Organizes indices of styled human variants within the batch into `batch_styled_human_groups` for efficient calculation of style alignment loss.

### Model Architecture

**`SAFDModel(nn.Module)`**:
* Defines the neural network architecture.
* Loads a pre-trained transformer model from `MODEL_NAME` using `AutoModel.from_pretrained()`.
* Optionally applies LoRA to the backbone model using `peft.get_peft_model()` if `USE_LORA` is `True`.
* Adds two linear heads on top of the transformer's pooled output:
    * `classification_head`: For Human/LLM classification.
    * `attribution_head`: For multi-label attribution prediction.
* `forward(input_ids, attention_mask)`: Passes inputs through the backbone, takes the last hidden state of the last token (common practice for sequence classification, though CLS token is also an option depending on the model), and then through the two heads to get `class_logits` and `attr_logits`.
* `save_lora_model(path)`: Saves LoRA adapter weights if `USE_LORA` is used.

### Loss Functions

**`classification_loss_fn(class_logits, class_labels)`**:
* Calculates Cross-Entropy Loss for the binary classification task.

**`attribution_loss_fn(attr_logits, attr_labels, valid_attr_samples_mask)`**:
* Calculates Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`) for the multi-label attribution task.
* Only applied to samples that are LLM-generated and have attribution labels (masked by `valid_attr_samples_mask`).

**`style_alignment_loss_fn(class_logits_groups)`**:
* Calculates a style alignment loss to encourage similar classification probability distributions for an original human text and its stylistic variants.
* Takes a list of `class_logits` tensors, where each tensor corresponds to a group of an original human text and its variants.
* For each group, it computes the sum of pairwise Kullback-Leibler (KL) divergences between the softmax-normalized `class_logits` of all pairs of samples within that group.
* The goal is to minimize this divergence, making the model's predictions for these related texts more consistent.
* The loss is normalized by the number of pairs and then averaged across all groups in the batch.

### Utility Functions

* **`load_and_split_data(json_path, test_size, random_state)`**: Loads data from a `.jsonl` file and splits it into training and testing sets, attempting stratified splitting if possible.
* **`save_data_to_jsonl(data, file_path)`**: Saves a list of data dictionaries to a `.jsonl` file.
* **`calculate_metrics(y_true, y_pred_proba, y_pred_labels)`**: Computes accuracy, precision, recall, F1-score, and AUROC. Handles cases with only one class present for AUROC.
* **`plot_confusion_matrix_util(y_true, y_pred_labels, class_names, save_filename)`**: Generates and saves a confusion matrix plot using Matplotlib.
* **`create_dummy_data_if_not_exists(dummy_data_path)`**: Creates a sample `.jsonl` data file if it doesn't already exist, useful for quick starts and understanding the data format.

### Training and Evaluation

**`train_epoch(model, dataloader, optimizer, scheduler, device)`**:
* Performs one epoch of training.
* Iterates through batches from the `dataloader`.
* Calculates `l_class`, `l_attr`, and `l_style`.
* Computes the total weighted loss: `` `ALPHA_CLASS * l_class + ALPHA_ATTR * l_attr + ALPHA_STYLE * l_style` ``.
* Performs backpropagation and optimizer step.
* Optionally steps the learning rate scheduler.
* Logs batch-level and epoch-average losses.

**`evaluate_model(model, dataloader, device)`**:
* Evaluates the model on the provided `dataloader` (typically the test set).
* Operates in `torch.no_grad()` mode.
* Collects all predictions and true labels.
* Calculates and returns evaluation metrics using `calculate_metrics`.
* Plots and saves the confusion matrix.

### Main Execution

**`main()`**:
* Sets up the device (`CUDA` or `CPU`).
* Generates/loads data, splits it, and saves the splits.
* Initializes tokenizer, datasets, and dataloaders.
* Initializes the `SAFDModel`, optimizer, and learning rate scheduler.
* Handles potential tokenizer `pad_token` issues.
* Manages training loop if `EPOCHS > 0`.
* Calls evaluation function if a test dataloader exists.
* Prints and saves final metrics.
* Saves the model components:
    * LoRA adapter (if `USE_LORA=True`).
    * Classification head state dictionary.
    * Attribution head state dictionary.
    * Tokenizer configuration and vocabulary.


### Output

Upon successful execution, the script will generate:

1.  **Model Files (in `MODEL_SAVE_PATH`, default: `safd_model_final/`)**:
    * If `USE_LORA=True`:
        * `lora_adapter/adapter_model.bin` (or `.safetensors`) and `adapter_config.json` (LoRA weights and config)
    * `classification_head.pth` (state dictionary of the classification head)
    * `attribution_head.pth` (state dictionary of the attribution head)
    * `tokenizer/` (directory containing tokenizer files: `tokenizer_config.json`, `vocab.json`, `merges.txt`, etc.)

2.  **Metrics File (`METRICS_SAVE_PATH`, default: `evaluation_metrics.json`)**:
    * A JSON file containing the evaluation metrics (accuracy, precision, recall, F1-score, AUROC).

3.  **Plots (`PLOTS_SAVE_PATH`, default: `evaluation_plots/`)**:
    * `confusion_matrix_test.png`: The confusion matrix for the test set predictions.

4.  **Data Splits**:
    * `TRAIN_SPLIT_PATH` (default: `train_split.jsonl`): The training data subset.
    * `TEST_SPLIT_PATH` (default: `test_split.jsonl`): The test data subset.

5.  **Dummy Data (if created)**:
    * `DUMMY_DATA_PATH` (default: `dummy_data.jsonl`): The generated sample data.

### LoRA Fine-Tuning

If `USE_LORA` is set to `True`:

* The script initializes the backbone model with a `LoraConfig`.
* Only the LoRA parameters and the final classification/attribution heads are trained. This significantly reduces the number of trainable parameters and memory requirements.
* During model saving, `model.save_lora_model()` is called to save the trained LoRA adapter. The full backbone model is not saved, only the adapter.
* To use the fine-tuned model later, you would load the base model (`MODEL_NAME`) and then load the LoRA adapter weights on top of it, along with the saved classification and attribution heads.

## Running the Script

1. Modify the configuration constants at the top of `SAFD.py` as needed (e.g., `MODEL_NAME`, `EPOCHS`, `BATCH_SIZE`, data paths).

2. Execute the script from your terminal:

    ```bash
    python scripts/SAFD.py
    ```