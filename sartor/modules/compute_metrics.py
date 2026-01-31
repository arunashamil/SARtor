import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
    return {"rouge2": float(scores["rouge2"])}
