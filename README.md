# Rocky Chatbot using T5

A generative AI project that uses an encoder-decoder model to train a chatbot to speak in the style of Rocky from Project Hail Mary.

## Overview

This project fine tunes T5 on data from the Project Hail Mary book and movie script to train the model to respond to prompts in the style of the character Rocky. I used question-answer pairs from the data that had a different character’s dialogue as the question and Rocky’s dialogue as the answer. I augmented the data so that there were multiple copies of Rocky’s response paired with differently worded questions.


## Dataset

**Project Hail Mary - Novel**
- ~150,000 words
- cleaned to 610 pairs

**Project Hail Mary - Movie**
- ~11,000 words
- cleaned to 201 pairs


## Model Architecture

### Finetuned T5
- Encoder-decoder transformer architecture

## Finetuning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 8 | Training batch size |
| Epochs | 5 | Total training epochs |
| Learning Rate | 5e-5 | AdamW optimizer learning rate |

### Repository Sections

1. **Data** - Original, processed, and cleaned, and augmented data
2. **Preprocessing** - Data preprocessing scripts
3. **Training** - Trains the model for 5 epochs with checkpointing
4. **Inference** - Performs inference on the model to interact
5. **GUI** - Graphic user interface to interact with model
6. **Conversation logs** - Conversation logs from selected chats

## Setup & Installation

1. Create a virtual environment:
```bash
conda create -n rocky
conda activate rocky
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training
```bash
train.ipynb
```

Set correct path for train and validation data and output directory for model weights.

Run all cells.

## Inference

```bash
inference.py
```

Run in terminal:

```bash
python inference.py --model <model-weight-path>
```

## Model Weights

Download model weights from: https://drive.google.com/drive/folders/1ZClfJ5rK4lAmMvdE4COgyxUrlHNlAMEc

| Name | Description |
|-----------------------|-------------------------|
| dialogue-augmented-t5 | t5 trained on 10x augmented dialogue-only training pairs|
| dialogue-t5 | t5 trained on dialogue-only training pairs |
| rocky-only-t5 | t5 trained on Rocky-only dialogue, with no question |
| rocky-t5 | t5 trained on Rocky dialogue with full paragraph context as question |

Recommended: dialogue-augmented-t5, checkpoint 5.

## GUI

```bash
gradio_app.py
```

Run in terminal:

```bash
python gradio_app.py --model <model-weight-path>
```