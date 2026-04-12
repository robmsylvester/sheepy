# Sheepy Examples

## Setup

```bash
# From the repo root
pip install -e ".[dev]"
```

## SMS Spam (Binary Classification)

```bash
cd examples
python main.py --config config/config_sms_spam.json --data_dir datasets/spam --output_dir output --time now
```

Or using the CLI:

```bash
sheepy train --config examples/config/config_sms_spam.json --data-dir examples/datasets/spam
```

## SemEval Sentiment (Multiclass)

```bash
python main.py --config config/config_semeval.json --output_dir output --time now
```

## Toxic Comments (Multilabel)

```bash
python main.py --config config/config_toxic_comments.json --data_dir datasets/toxic_comments --output_dir output --time now
```

## Configuration

See `config/config_guide.json` for a reference of all config keys.
All configs support JSON, YAML, and TOML formats.
