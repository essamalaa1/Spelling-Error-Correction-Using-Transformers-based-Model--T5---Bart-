# TensorFlow Spelling Correction with T5 and BART

This project demonstrates how to fine-tune T5 and BART transformer models for spelling correction tasks using TensorFlow and the Hugging Face `transformers` library. The primary code is contained within the `Model Training Notebook.ipynb` Jupyter notebook.

## Overview

The notebook covers the entire pipeline for training spelling correction models:
1.  **Data Loading & Preprocessing**: Loads sentences from TSV files, cleans them by removing punctuation and digits.
2.  **Synthetic Misspelling Generation**: Introduces various character-level errors (deletion, insertion, substitution, transposition) into correct sentences to create misspelled-correct pairs for training.
3.  **Model Training (T5)**:
    *   Uses `t5-small` as the base model.
    *   Tokenizes data with a "fix spelling: " prefix.
    *   Fine-tunes the model on the generated dataset.
    *   Evaluates using Word Error Rate (WER) and Character Error Rate (CER).
    *   Saves the fine-tuned T5 model and tokenizer.
4.  **Model Training (BART)**:
    *   Trains a `facebook/bart-base` model from scratch on the spelling correction task.
    *   Fine-tunes a pre-trained spelling correction model (`oliverguhr/spelling-correction-english-base`) on the custom dataset.
    *   Evaluates both BART models using WER and CER.
5.  **Inference**: Shows examples of how to use the trained models for correcting misspelled sentences.

## Features

*   Data loading and cleaning for text-based datasets.
*   Customizable synthetic misspelling generation.
*   Fine-tuning of T5 (e.g., `t5-small`).
*   Fine-tuning of BART (e.g., `facebook/bart-base` and `oliverguhr/spelling-correction-english-base`).
*   Evaluation with standard ASR/NLP metrics: WER & CER.
*   Saving and loading of trained models and tokenizers.
*   TensorFlow-based training pipeline.

## Models Used

*   **T5-small**: `t5-small`
*   **BART-base**: `facebook/bart-base`
*   **BART (pre-trained for spelling)**: `oliverguhr/spelling-correction-english-base`

## Dataset

The notebook expects three TSV files:
*   `tune.tsv`: For generating training data.
*   `validation.tsv`: For generating validation data.
*   `test.tsv`: For generating test data.

Each file should contain one sentence per line in the first column.
The notebook preprocesses these sentences by:
*   Stripping leading/trailing quotes and spaces.
*   Removing all punctuation and digits.
*   Normalizing whitespace.

Misspelled versions are then generated programmatically for training, validation, and testing.

**Place your data files in a `data/` directory at the root of this project:**


If you use a different path, modify the `os.path.join("/content/data/", ...)` parts in the notebook accordingly.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tensorflow-spelling-correction-transformer.git
    cd tensorflow-spelling-correction-transformer
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The notebook installs the following. You can install them directly using pip:
    ```bash
    pip install datasets jiwer numpy pandas tensorflow transformers nltk
    ```
    The notebook uses `tf_keras` for optimizer and callbacks in some places. Ensure your TensorFlow version is compatible or adjust imports to `tensorflow.keras` if needed. The notebook was run with TensorFlow 2.18.0.

4.  **Download NLTK 'punkt' tokenizer:**
    The notebook includes a check and download for `nltk.download('punkt', quiet=True)`. If running outside the notebook or if it fails, you might need to run this in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5.  **GPU (Recommended for Training):**
    Ensure you have a compatible GPU and an appropriate version of CUDA/cuDNN installed if you want to train models efficiently. The notebook is configured to run on a GPU if available.

## Usage

1.  **Prepare your dataset**: Place `tune.tsv`, `validation.tsv`, and `test.tsv` in the `data/` directory as described above.
2.  **Open and run the Jupyter Notebook**:
    ```bash
    jupyter notebook "Model Training Notebook.ipynb"
    ```
    Execute the cells sequentially.

### Key Parameters in the Notebook

You can adjust several parameters within the notebook cells:
*   `MAX_SENTENCES_TUNE`, `MAX_SENTENCES_VAL_TEST`: To limit the number of sentences used from each dataset (set to `None` to use all).
*   `VERSIONS_PER_SENTENCE`: Number of misspelled versions to generate per correct sentence for the training set.
*   `MODEL_NAME` (for T5 and initial BART), `MODEL_CHECKPOINT` (for pre-trained BART): To change the base models.
*   `MAX_INPUT_LENGTH`, `MAX_TARGET_LENGTH`: Tokenizer maximum sequence lengths.
*   `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`: Training hyperparameters.
*   `SAVE_DIRECTORY`: Path where the fine-tuned T5 model and tokenizer will be saved.

## Evaluation Results (from the notebook)

The notebook evaluates the models on the generated test set.
*   **Fine-tuned T5-small:**
    *   WER: ~11.524%
    *   CER: ~5.248%
*   **BART-base (trained from scratch on task, evaluated on 10 test batches):**
    *   WER: ~6.288%
    *   CER: ~3.572%
*   **BART (fine-tuned from `oliverguhr/spelling-correction-english-base`, evaluated on 10 test batches):**
    *   WER: ~3.877%
    *   CER: ~1.354%

*Note: The BART model evaluations were performed on a subset (10 batches) of the test data for quicker demonstration. Full evaluation might yield slightly different results.*

## Saved T5 Model

The fine-tuned T5 model and its tokenizer are saved to the directory specified by `SAVE_DIRECTORY` (default: `./my_spell_corrector_t5_small`). You can load this model for later use.

## License

This project is unlicensed. Feel free to use, modify, and distribute as you see fit.
