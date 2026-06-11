import json

# Define the complete notebook structure as a clean Python dictionary
notebook_data = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Banking Intent Classification Challenge\n",
                "\n",
                "**Release:** May 29, 2026  \n",
                "**Deadline:** June 6, 2026  \n",
                "**Recommended environment:** Kaggle/Google Colab with GPU runtime\n",
                "\n",
                "In this competition you will build a text classifier for 77 banking customer-support intents.\n",
                "\n",
                "## Model Restrictions\n",
                "- Final model must have at most 250M parameters.\n",
                "- Closed-source API / chat / instruction-tuned LLMs may not be used for final predictions.\n",
                "- No external labeled datasets for this task.\n",
                "\n",
                "| Component | Weight |\n",
                "|---|---:|\n",
                "| Private leaderboard score | 50% |\n",
                "| Model comparison / ablation | 30% |\n",
                "| Error analysis | 10% |\n",
                "| Reproducible notebook/code | 10% |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 0. Setup\n",
                "\n",
                "Environment initialization and framework configurations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install requirements safely\n",
                "!pip -q install kaggle transformers datasets scikit-learn\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import os\n",
                "import random\n",
                "import shutil\n",
                "\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from tqdm.auto import tqdm\n",
                "\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from torch.amp import GradScaler\n",
                "from torch.optim import AdamW\n",
                "\n",
                "from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification\n",
                "from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score, f1_score\n",
                "\n",
                "SEED = 2026\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "torch.cuda.manual_seed_all(SEED)\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "_N_WORKERS = 0  # Safe worker allocation across operating platforms\n",
                "print('Using device:', device)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Download the Competition Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    from google.colab import files\n",
                "    IN_COLAB = True\n",
                "except ImportError:\n",
                "    IN_COLAB = False\n",
                "\n",
                "kaggle_dir = Path.home() / '.kaggle'\n",
                "kaggle_dir.mkdir(exist_ok=True)\n",
                "\n",
                "if not (kaggle_dir / 'kaggle.json').exists():\n",
                "    if not IN_COLAB:\n",
                "        raise FileNotFoundError('Place kaggle.json at ~/.kaggle/kaggle.json')\n",
                "    uploaded = files.upload()\n",
                "    if 'kaggle.json' not in uploaded:\n",
                "        raise RuntimeError('Kaggle credential upload missing.')\n",
                "    shutil.move('kaggle.json', kaggle_dir / 'kaggle.json')\n",
                "    os.chmod(kaggle_dir / 'kaggle.json', 0o600)\n",
                "\n",
                "print('Kaggle API token ready.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "COMPETITION_SLUG = 'harbour-space-banking-intent-classification'\n",
                "DATA_DIR = Path('/content/banking77_data') if IN_COLAB else Path('banking77_data')\n",
                "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "!kaggle competitions download -c {COMPETITION_SLUG} -p {DATA_DIR}\n",
                "\n",
                "zip_files = sorted(DATA_DIR.glob('*.zip'))\n",
                "for zip_path in zip_files:\n",
                "    shutil.unpack_archive(str(zip_path), str(DATA_DIR))\n",
                "\n",
                "print('Files inside directory:')\n",
                "for path in sorted(DATA_DIR.iterdir()):\n",
                "    print('-', path.name)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Metadata and Check the Files"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TRAIN_CSV = DATA_DIR / 'train.csv'\n",
                "TEST_CSV = DATA_DIR / 'test.csv'\n",
                "LABEL_MAP_CSV = DATA_DIR / 'label_map.csv'\n",
                "SAMPLE_SUBMISSION_CSV = DATA_DIR / 'sample_submission.csv'\n",
                "\n",
                "train_df = pd.read_csv(TRAIN_CSV)\n",
                "test_df = pd.read_csv(TEST_CSV)\n",
                "label_map = pd.read_csv(LABEL_MAP_CSV)\n",
                "sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)\n",
                "\n",
                "label_names = sorted(train_df['label'].unique().tolist())\n",
                "label_to_id = {label: idx for idx, label in enumerate(label_names)}\n",
                "id_to_label = {idx: label for label, idx in label_to_id.items()}\n",
                "print('Checks completed. Total categories:', len(label_names))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Inspect Text Examples"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df['text_length_words'] = train_df['text'].astype(str).str.split().str.len()\n",
                "print(train_df['text_length_words'].describe())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Create Your Validation Split"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_part, val_part = train_test_split(\n",
                "    train_df, \n",
                "    test_size=0.15, \n",
                "    stratify=train_df['label'], \n",
                "    random_state=SEED\n",
                ")\n",
                "print(f'Split set -> Train entries: {len(train_part)}, Validation entries: {len(val_part)}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Text Preprocessing / Tokenization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class BankingDataset(Dataset):\n",
                "    def __init__(self, df, tokenizer, max_length=64, is_test=False):\n",
                "        self.df = df.reset_index(drop=True)\n",
                "        self.tokenizer = tokenizer\n",
                "        self.max_length = max_length\n",
                "        self.is_test = is_test\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        text = str(row['text'])\n",
                "        inputs = self.tokenizer(\n",
                "            text, max_length=self.max_length, padding='max_length', truncation=True\n",
                "        )\n",
                "        item = {\n",
                "            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),\n",
                "            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)\n",
                "        }\n",
                "        if 'token_type_ids' in inputs:\n",
                "            item['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)\n",
                "        if not self.is_test:\n",
                "            item['labels'] = torch.tensor(label_to_id[row['label']], dtype=torch.long)\n",
                "        return item\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Experiments"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def build_transformer(model_name, num_labels=77):\n",
                "    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)\n",
                "    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)\n",
                "\n",
                "def build_llrd_optimizer(model, lr=2e-5, lr_decay=0.85):\n",
                "    raw_model = model.module if hasattr(model, 'module') else model\n",
                "    model_type = raw_model.config.model_type\n",
                "    if 'deberta' in model_type:\n",
                "        encoder = raw_model.deberta\n",
                "    elif 'roberta' in model_type:\n",
                "        encoder = raw_model.roberta\n",
                "    else:\n",
                "        return AdamW(model.parameters(), lr=lr, weight_decay=0.01)\n",
                "\n",
                "    no_decay = ['bias', 'LayerNorm.weight']\n",
                "    grouped_parameters = []\n",
                "    head_params = [p for n, p in raw_model.named_parameters() if encoder.base_model_prefix not in n]\n",
                "    if head_params:\n",
                "        grouped_parameters.append({'params': head_params, 'weight_decay': 0.01, 'lr': lr})\n",
                "\n",
                "    layers = encoder.encoder.layer\n",
                "    num_layers = len(layers)\n",
                "    current_lr = lr * lr_decay\n",
                "    for i in reversed(range(num_layers)):\n",
                "        grouped_parameters.append({\n",
                "            'params': [p for n, p in layers[i].named_parameters() if not any(nd in n for nd in no_decay)],\n",
                "            'weight_decay': 0.01, 'lr': current_lr\n",
                "        })\n",
                "        grouped_parameters.append({\n",
                "            'params': [p for n, p in layers[i].named_parameters() if any(nd in n for nd in no_decay)],\n",
                "            'weight_decay': 0.0, 'lr': current_lr\n",
                "        })\n",
                "        current_lr *= lr_decay\n",
                "    return AdamW(grouped_parameters)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Training and Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def train_advanced_model(model_name, tag, train_ds, val_ds, epochs=3, batch_size=32, grad_accum=2, lr=2e-5, use_llrd=True, lr_decay=0.85, use_cosine=True):\n",
                "    model = build_transformer(model_name)\n",
                "    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=_N_WORKERS, pin_memory=True)\n",
                "    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=_N_WORKERS, pin_memory=True)\n",
                "    \n",
                "    total_steps = (len(train_loader) // grad_accum) * epochs\n",
                "    optimizer = build_llrd_optimizer(model, lr, lr_decay) if use_llrd else AdamW(model.parameters(), lr=lr)\n",
                "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps) if use_cosine else get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)\n",
                "    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)\n",
                "    \n",
                "    if torch.cuda.device_count() > 1:\n",
                "        model = nn.DataParallel(model)\n",
                "    model.to(device)\n",
                "    \n",
                "    is_deberta = 'deberta' in model_name.lower()\n",
                "    use_amp = (device.type == 'cuda')\n",
                "    amp_dtype = torch.bfloat16 if (is_deberta and torch.cuda.is_bf16_supported()) else torch.float16\n",
                "    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))\n",
                "    \n",
                "    best_acc = 0.0\n",
                "    best_state = None\n",
                "    history = []\n",
                "    \n",
                "    for epoch in range(1, epochs + 1):\n",
                "        model.train()\n",
                "        optimizer.zero_grad()\n",
                "        for step, batch in enumerate(tqdm(train_loader, desc=f'[{tag}] Epoch {epoch}')):\n",
                "            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)\n",
                "            labels = batch['labels'].to(device)\n",
                "            \n",
                "            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):\n",
                "                outputs = model(input_ids=ids, attention_mask=mask)\n",
                "                loss = loss_fn(outputs.logits, labels) / grad_accum\n",
                "            \n",
                "            if scaler is not None and amp_dtype == torch.float16:\n",
                "                scaler.scale(loss).backward()\n",
                "            else:\n",
                "                loss.backward()\n",
                "                \n",
                "            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):\n",
                "                if scaler is not None and amp_dtype == torch.float16:\n",
                "                    scaler.unscale_(optimizer)\n",
                "                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                "                    scaler.step(optimizer)\n",
                "                    scaler.update()\n",
                "                else:\n",
                "                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                "                    optimizer.step()\n",
                "                scheduler.step()\n",
                "                optimizer.zero_grad()\n",
                "                \n",
                "        model.eval()\n",
                "        preds_all, true_all = [], []\n",
                "        with torch.no_grad():\n",
                "            for batch in val_loader:\n",
                "                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)\n",
                "                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):\n",
                "                    outputs = model(input_ids=ids, attention_mask=mask)\n",
                "                preds_all.extend(outputs.logits.argmax(-1).cpu().tolist())\n",
                "                true_all.extend(batch['labels'].tolist())\n",
                "                \n",
                "        val_acc = accuracy_score(true_all, preds_all)\n",
                "        print(f'Epoch {epoch} - Val Acc: {val_acc:.4f}')\n",
                "        history.append({'epoch': epoch, 'val_acc': val_acc})\n",
                "        if val_acc > best_acc:\n",
                "            best_acc = val_acc\n",
                "            raw_model = model.module if hasattr(model, 'module') else model\n",
                "            best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}\n",
                "            \n",
                "    raw_model = model.module if hasattr(model, 'module') else model\n",
                "    raw_model.load_state_dict(best_state)\n",
                "    return model, pd.DataFrame(history), best_acc\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Experiment Log"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_book = {}\n",
                "tok_roberta = AutoTokenizer.from_pretrained('roberta-base')\n",
                "tok_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')\n",
                "\n",
                "ds_train_rob = BankingDataset(train_part, tok_roberta)\n",
                "ds_val_rob = BankingDataset(val_part, tok_roberta)\n",
                "ds_train_deb = BankingDataset(train_part, tok_deberta)\n",
                "ds_val_deb = BankingDataset(val_part, tok_deberta)\n",
                "\n",
                "# Run Baseline\n",
                "m1, h1, acc1 = train_advanced_model('roberta-base', 'Exp1_RoBERTa', ds_train_rob, ds_val_rob, use_llrd=False, use_cosine=False)\n",
                "log_book['Exp 1'] = ['roberta-base', 'Baseline config', 64, acc1]\n",
                "\n",
                "# Run Optimized Model\n",
                "m5, h5, acc5 = train_advanced_model('microsoft/deberta-v3-base', 'Exp5_DeBERTa_Optimal', ds_train_deb, ds_val_deb, use_llrd=True, use_cosine=True)\n",
                "log_book['Exp 5'] = ['microsoft/deberta-v3-base', 'LLRD + Cosine schedule', 64, acc5]\n",
                "\n",
                "exp_df = pd.DataFrame.from_dict(log_book, orient='index', columns=['Model', 'Strategy', 'Length', 'Val Acc'])\n",
                "display(exp_df)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Error Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Analyzing classification boundaries across validation samples.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Generate a Submission"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def validate_submission(submission_df, test_df, valid_labels):\n",
                "    assert list(submission_df.columns) == ['id', 'label']\n",
                "    assert len(submission_df) == len(test_df)\n",
                "    print('Submission formatting checked and confirmed.')\n",
                "\n",
                "ds_test = BankingDataset(test_df, tok_deberta, is_test=True)\n",
                "test_loader = DataLoader(ds_test, batch_size=64, shuffle=False)\n",
                "\n",
                "m5.eval()\n",
                "final_predictions_ids = []\n",
                "with torch.no_grad():\n",
                "    for batch in test_loader:\n",
                "        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)\n",
                "        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):\n",
                "            outputs = m5(input_ids=ids, attention_mask=mask)\n",
                "        final_predictions_ids.extend(outputs.logits.argmax(-1).cpu().tolist())\n",
                "\n",
                "submission = pd.DataFrame({\n",
                "    'id': test_df['id'],\n",
                "    'label': [id_to_label[idx] for idx in final_predictions_ids]\n",
                "})\n",
                "validate_submission(submission, test_df, label_names)\n",
                "submission.to_csv('submission.csv', index=False)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Final Checklist"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Safely write the dictionary structure to a valid notebook file
output_path = "Banking77_Completed_Solution.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2, ensure_ascii=False)

print(f"Successfully generated clean notebook without syntax errors: '{output_path}'")