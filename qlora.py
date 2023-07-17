import argparse
import json
import logging
import os
import sys
from os.path import join

import numpy as np
import torch
import transformers
import wandb
from model.dataclasses import DataArguments, ModelArguments, TrainingArguments
from model.helpers import (SavePeftModelCallback, get_accelerate_model,
                           get_last_checkpoint, print_trainable_parameters)
from model.merge import merge_weights
from model.preprocess import (make_data_module,
                              smart_tokenizer_and_embedding_resize)
from transformers import (AutoTokenizer, LlamaTokenizer, Seq2SeqTrainer,
                          set_seed)

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    finetune_id = args.finetune_id
    truffle_size = args.truffle_size

    if completed_training:
        print('Detected that training was already completed!')
        merge_weights(
            join(checkpoint_dir, "adapter_model"),
            join(checkpoint_dir, "merged"),
            args.model_name_or_path
        )
        sys.exit(123)

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    with wandb.init(
            project=f"truffle-{truffle_size}",
            id=finetune_id,
            resume="allow",) as run:
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False,  # Fast tokenizer giving issues.
            # Needed for HF name change
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
            use_auth_token=args.use_auth_token,
        )
        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
            # LLaMA tokenizer may not have correct special tokens set.
            # Check and add them if missing to prevent them from being parsed into different tokens.
            # Note that these are present in the vocabulary.
            # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print('Adding special tokens.')
            tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -
                    1 else tokenizer.pad_token_id
                ),
            })
        data_module = make_data_module(tokenizer=tokenizer, args=args)
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
        )

        # Callbacks
        if not args.full_finetune:
            trainer.add_callback(SavePeftModelCallback)

        # Verifying the datatypes.
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v/total)

        all_metrics = {"run_name": args.run_name}
        # Training
        if args.do_train:
            logger.info("*** Train ***")
            # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
            # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
            train_result = trainer.train(checkpoint_dir)
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            all_metrics.update(metrics)
        # Evaluation
        if args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            all_metrics.update(metrics)
        # Prediction
        if args.do_predict:
            logger.info("*** Predict ***")
            prediction_output = trainer.predict(
                test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
            prediction_metrics = prediction_output.metrics
            predictions = prediction_output.predictions
            predictions = np.where(predictions != -100,
                                   predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
                for i, example in enumerate(data_module['predict_dataset']):
                    example['prediction_with_input'] = predictions[i].strip()
                    example['prediction'] = predictions[i].replace(
                        example['input'], '').strip()
                    fout.write(json.dumps(example) + '\n')
            print(prediction_metrics)
            trainer.log_metrics("predict", prediction_metrics)
            trainer.save_metrics("predict", prediction_metrics)
            all_metrics.update(prediction_metrics)

        if (args.do_train or args.do_eval or args.do_predict):
            with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
                fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
