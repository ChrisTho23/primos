# Primos
This project aims to investigate whether task-specific fine-tuning can significantly enhance the performance of foundational large language models (LLMs) on complex, non-trivial natural language processing (NLP) tasks relevant to businesses. The approach involves using QLoRA, a parameter-efficient fine-tuning method, to fine-tune an open-source model on a synthetic task that mimics a real-world problem. The performance of the fine-tuned model will be compared with that of non-fine-tuned open- and closed-source state-of-the-art models.

## Overview
This project is still in progress. Below are the steps that have been completed so far:

1. **Dataset Generation**: A dataset has been generated to mimic a non-trivial task relevant in the real world that could be solved by an LLM. Here, I decided to model a task faced by customer support departments. Specifically, it represents support tickets raised by customers of a medium-sized German electronics company (in German). These tickets need to be categorized accurately and efficiently to ensure prompt and appropriate responses. The key challenge is to classify each ticket into the correct category, subcategory, and urgency level for efficient downstream processing. This task involves understanding the nuances of customer queries, which can vary widely in language and detail, and assigning them to predefined categories and subcategories for further processing. An accurate classification system can significantly enhance the efficiency of customer support operations, reduce response times, and improve customer satisfaction. The task is designed to be challenging enough for foundational models to perform poorly on it. First, I use OpenAI's GPT4o model to syntetically generate a relational database as it could exist in a company. The database contains the following keys: title, description, user, date, category, subcategory, and urgency. An example of the dataset is provided below. Currently, for testing purposes, the dataset contains only 110 samples spanning from 5 categories and 10 subcategories. I also include 10 samples that are not related to the task and should be classified as not relevant.

    Example:
    ```json
    {
        "title": "Verspätete Lieferung",
        "description": "Meine Bestellung sollte vor einer Woche ankommen, aber sie ist immer noch nicht da. Können Sie den Lieferstatus überprüfen? Meine Bestellnummer ist 54321.",
        "user": "Michael König",
        "date": "2024-05-26",
        "category": "Bestellverwaltung",
        "subcategory": "Lieferverzögerungen",
        "urgency": "Mittel"
    }
    ```

2. **Instruction Dataset Creation**: The database has been transformed into an instruction dataset using ChatML to perform task-specific fine-tuning.
    ```
    [
        {
            'role': 'user',
            'content': "You are an AI model trained to categorize customer support tickets for a German consumer electronics company. Your task is to determine the most appropriate category and subcategory for the support ticket provided below, and also classify the urgency of the ticket.\n\nProvide the result in a JSON format with the following fields:\n- **category**: The main category of the ticket\n- **subcategory**: The subcategory of the ticket\n- **urgency**: The urgency level of the ticket\n\nThe possible categories, subcategories, and urgency levels are as follows:\n\n**Categories and subcategories**:\n**Technischer Support**\n- Geräte-Setup-Probleme\n- Softwarefehler\n\n**Abrechnung und Zahlungen**\n- Zahlungsprobleme\n- Rückerstattungsanfragen\n\n**Produktinformationen**\n- Produktspezifikationen\n- Garantieinformationen\n\n**Bestellverwaltung**\n- Bestellverfolgung\n- Lieferverzögerungen\n\n**Allgemeine Anfragen**\n- Unternehmensrichtlinien\n- Feedback und Vorschläge\n\n**Ungewiss**\n- Kein Zusammenhang\n\n**Urgency Levels**:\n    - Hoch\n    - Mittel\n    - Niedrig\n=== Support Ticket ===\nTitle: Smartphone erkennt SIM-Karte nicht\nDescription: Ich habe das neue SmartX Ultra gekauft und beim Einrichten erkennt das Smartphone meine SIM-Karte nicht. Es zeigt ständig 'Keine SIM-Karte'. Ich habe bereits verschiedene SIM-Karten ausprobiert, aber das Problem bleibt bestehen.\nName: Laura Schmidt\nDate: 2024-06-01\n"
        }
        {
            'role': 'assistant',
            'content': "{'category': 'Technischer Support', 'subcategory': 'Geräte-Setup-Probleme', 'urgency': 'Hoch'}"
        }
    ]
    ```

3. **Benchmarking Foundational Models**: The performance of state-of-the-art foundational models (OpenAI's GPT3.5, GPT4, and Mistral 7B) has been benchmarked on a hold-out set consisting of 10% of the dataset (11 samples). The results are summarized in the table below. (Please fill in the accuracy values for each model.)

    ### Accuracy Comparison
    | Model            | Category Accuracy | Subcategory Accuracy | Urgency Accuracy |
    |------------------|-------------------|----------------------|------------------|
    | **GPT-3.5**      |0.7                   |0.65                      |0.53                  |
    | **GPT-4**        |0.75                   |0.72                      |0.51                  |
    | **Mistral 7B**   |0.8                   |0.83                      |0.64                  |

4. **Task-Specific Fine-Tuning**: Task-specific fine-tuning has been performed on an open-source LLM using the QLoRA framework. The model is first double-quantized (weights to 4-bit NF4, and first-level constants are also quantized), and then LoRA is set up on all attention and parts of the feed-forward layer (o_proj, gate_proj) with alpha = r = 8. Finally, LoRA is trained for 1 epoch with a cosine learning rate scheduler on the training set. The training is conducted on a cluster of two Nvidia L4 GPUs (24GB VRAM each, with approximately 5GB used for tuning). Currently, fine-tuning has been performed on Mistral 7B for testing purposes, with plans to run on Mistral 70B and LLama 3 70B. [Link to adapter](https://huggingface.co/ChrisTho/bizztune_mistral_7b_instruct)

5. **Benchmarking Fine-Tuned Model**: The fine-tuned model will be benchmarked against the foundational model on the hold-out validation set. (Please fill in the results in the table below. It is expected that performance will improve significantly for larger models and once input masking is introduced.)

    ### Fine-Tuned Model Comparison
    | Model            | Category Accuracy | Subcategory Accuracy | Urgency Accuracy |
    |------------------|-------------------|----------------------|------------------|
    | **Mistral 7B**   |0.73 (+3%)                   |0.64 (-0.7%)                      |0.64 (+20%)                  |

The end-to-end fine-tuning and evaluation pipeline can be executed by running `bizztune/main.py`. Note that at least 12GB of VRAM should be available to perform these tasks.

## Next Steps
The following steps are planned for the future:
- Fine-tuning LLama3 and Mistral 70GB using Fully Sharded Data Parallel (FSDP) (already implemented).
- Exploring performance differences when making all three classifications at once vs. only once per LLM call.
- Hyperparameter tuning.
- Additional fine-tuning experiments with larger datasets and different frameworks.
- Implementation of input masking for more accurate tokenization.

## Project Structure
```bash
├── .venv/
├── archive/
├── bizztune/
│   ├── __pycache__/
│   ├── baseset/
│   ├── config/
│   ├── instructionset/
│   ├── tune/
│   ├── main.py
│   ├── utils.py
├── data/
├── dist/
├── model/
├── tests/
├── .env
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Setting Up the Project
Clone the repository:
```bash
git clone https://github.com/ChrisTho23/bizztune.git
cd bizztune
```

Install dependencies using Poetry:
```bash
poetry install
```

Set up the environment variables:
Create a .env file in the root directory and add your API keys for Mistral and OpenAI.
```bash
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Running Features
1. **Create Dataset**
    To create the dataset, run the following Poetry script:
    ```bash
    poetry run create_dataset
    ```
    This will execute the main function in bizztune.data.

2. **Evaluate Dataset with GPT & Mistral**
    To evaluate the dataset, run the following Poetry script:
    ```bash
    poetry run benchmark
    ```
    This will execute the main function in bizztune.benchmark.

## Configuration
The configurations for each task can be found in bizztune/config.py. Adjust these configurations as needed for your specific requirements.
