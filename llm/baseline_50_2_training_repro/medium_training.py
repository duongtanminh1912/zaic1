"""Medium V3 ASR branch specialized from the archived trainer.
No model loading or training on import; reproduce.py binds verified paths/guard.
See PROVENANCE.json for source hash and the repeated-save callback repair.
"""
from contextlib import nullcontext
import gc, hashlib, importlib.metadata, json, math, os, random, sys, time
from pathlib import Path
from train_core.artifact_io import sha, new, json_new
from train_core.asr_training_policy import words, distance
HERE = BASE_DIR = RAW_TRAIN = None

def verify(data, full=False):
    raise RuntimeError('Use reproduce.py to bind the verified data guard')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def atomic(p, d):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    t.replace(p)

def gpu_setup():
    import numpy as np
    import torch
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported(), 'Local BF16 training needs a natively supported GPU'
    torch.cuda.set_per_process_memory_fraction(0.95, 0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return (torch, np)

def asr_collator(processor, decoder_start, torch):

    def collate(rows):
        labels = processor.tokenizer.pad([{'input_ids': r['labels']} for r in rows], return_tensors='pt')
        y = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
        if (y[:, 0] == decoder_start).all().item():
            y = y[:, 1:]
        return {'input_features': torch.stack([r['input_features'] for r in rows]), 'attention_mask': torch.stack([r['attention_mask'] for r in rows]), 'labels': y}
    return collate

def asr_dataset(clips, processor, torch, np):
    import wave
    base = RAW_TRAIN

    class Dataset(torch.utils.data.Dataset):

        def __len__(self):
            return len(clips)

        def __getitem__(self, index):
            row = clips[index]
            p = (base / row['audio']).resolve()
            assert p.is_relative_to((base / 'audio').resolve())
            with wave.open(str(p), 'rb') as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
                values = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768
            features = processor.feature_extractor(values, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            return {'input_features': features.input_features[0], 'attention_mask': features.attention_mask[0], 'labels': processor.tokenizer(row['text']).input_ids}
    return Dataset()

def evaluate_asr(model, processor, clips, torch, np, output, step):
    ds = asr_dataset(clips, processor, torch, np)
    was_training = model.training
    model.eval()
    predictions = []
    edits = denominator = failed = 0
    started = time.monotonic()
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for j, row in enumerate(clips):
            item = ds[j]
            generated = model.generate(input_features=item['input_features'][None].to('cuda', dtype=torch.bfloat16), attention_mask=item['attention_mask'][None].to('cuda'), language='vi', task='transcribe', do_sample=False, num_beams=1, max_new_tokens=256, return_timestamps=False, return_dict_in_generate=True, use_cache=True)
            ids = generated.sequences[0].tolist()
            truncated = not bool(eos_set.intersection(ids))
            text = '' if truncated else processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
            failed += int(truncated)
            a, b = (words(row['text']), words(text))
            edits += distance(a, b)
            denominator += len(a)
            predictions.append({'audio': row['audio'], 'sha256': row['sha256'], 'text': text, 'error': 'token_cap_without_eos' if truncated else None})
            if (j + 1) % 100 == 0:
                print('ASR_VAL', step, j + 1, '/', len(clips), flush=True)
    if was_training:
        model.train()
    result = {'step': step, 'n': len(clips), 'wer_percent': 100 * edits / denominator, 'edits': edits, 'reference_words': denominator, 'seconds': time.monotonic() - started, 'failed_samples': failed, 'predictions': predictions}
    json_new(output / f'VAL_ASR_STEP_{step}.json', result)
    return result

def training(data, output, smoke, resume):
    kind = 'asr'
    manifest = verify(data, full=not smoke)
    cfg = manifest['config'][kind]
    torch, np = gpu_setup()
    from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers.trainer_utils import get_last_checkpoint
    output.mkdir(parents=True, exist_ok=True)
    run = output / (kind + ('_smoke' if smoke else '_full'))
    run.mkdir(exist_ok=True)
    previous = read(run / 'RUN_MANIFEST.json') if (run / 'RUN_MANIFEST.json').exists() else None
    common = {'data_sha256': sha(data / 'DATA_MANIFEST.json'), 'source_sha256': sha(Path(__file__)), 'kind': kind, 'smoke': smoke, 'config': cfg, 'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')}}
    if previous:
        assert previous['configuration'] == common, 'Training configuration changed'
        if previous['status'] == 'completed':
            print('ALREADY_COMPLETED', kind, smoke, flush=True)
            return
        if not resume:
            raise RuntimeError('Interrupted run exists; explicitly resume from last saved optimizer checkpoint')
    else:
        atomic(run / 'RUN_MANIFEST.json', {'status': 'preparing', 'configuration': common})
    checkpoint = get_last_checkpoint(str(run / 'checkpoints')) if resume and (run / 'checkpoints').exists() else None
    if resume and previous and (not checkpoint):
        raise RuntimeError('No saved training checkpoint; preserve interrupted artifacts')
    folder = BASE_DIR
    processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language='vi', task='transcribe', predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(folder, local_files_only=True, use_safetensors=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = 'vi'
    model.generation_config.task = 'transcribe'
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'], target_modules=cfg['targets'], bias='none'))
    clips = read(data / 'train_asr_clips.json')
    if smoke:
        clips = sorted(clips, key=lambda r: len(words(r['text'])), reverse=True)[:2]
    train_rows = asr_dataset(clips, processor, torch, np)
    collator = asr_collator(processor, model.config.decoder_start_token_id, torch)
    eval_rows = None
    model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(('lora_' in n and 'visual' not in n for n, p in trainable))
    params = sum((p.numel() for n, p in trainable))
    initial_b = {n: p.detach().cpu().clone() for n, p in trainable if '.lora_B.' in n}
    args = dict(output_dir=str(run / 'checkpoints'), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1 if smoke else cfg['gradient_accumulation'], num_train_epochs=cfg['epochs'], learning_rate=cfg['learning_rate'], warmup_steps=cfg['warmup_steps'], weight_decay=cfg['weight_decay'], lr_scheduler_type='linear', bf16=True, fp16=False, optim='adamw_torch', gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1, logging_nan_inf_filter=False, torch_empty_cache_steps=None, save_steps=1 if smoke else cfg['save_steps'], save_strategy='steps', save_total_limit=2, eval_strategy='no', remove_unused_columns=False, dataloader_num_workers=0, report_to='none', seed=20261103, data_seed=20261103, max_grad_norm=1.0, push_to_hub=False, prediction_loss_only=True)
    if smoke:
        args['max_steps'] = 2
    training_args = TrainingArguments(**args)
    started = time.monotonic()
    val_history = read(run / 'VAL_HISTORY.json') if resume and (run / 'VAL_HISTORY.json').exists() else []
    handled_saves = set()

    class Progress(TrainerCallback):

        def on_step_end(self, args, state, control, **kwargs):
            atomic(run / 'PROGRESS.json', {'step': state.global_step, 'total_steps': state.max_steps, 'elapsed_s': time.monotonic() - started, 'cuda_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
            if state.global_step >= state.max_steps:
                control.should_save = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.global_step in handled_saves:
                return control
            handled_saves.add(state.global_step)
            saved = run / 'checkpoints' / f'checkpoint-{state.global_step}'
            required = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth']
            assert all(((saved / n).is_file() for n in required)), ('Incomplete resume checkpoint', str(saved))
            atomic(run / 'LATEST_CHECKPOINT.json', {'step': state.global_step, 'path': str(saved), 'files_sha256': {n: sha(saved / n) for n in required}})
            if not smoke and (abs(state.epoch - round(state.epoch)) < 1e-06 or state.global_step >= state.max_steps):
                epoch_dir = run / 'epoch_adapters' / f'step-{state.global_step}'
                assert not epoch_dir.exists(), 'Refuse overwrite epoch adapter'
                model.save_pretrained(epoch_dir, safe_serialization=True)
                processor.save_pretrained(epoch_dir / 'processor')
                json_new(epoch_dir / 'WEIGHTS_MANIFEST.json', {'step': state.global_step, 'epoch': state.epoch, 'adapter_sha256': sha(epoch_dir / 'adapter_model.safetensors')})
            if not smoke and (abs(state.epoch - round(state.epoch)) < 1e-06 or state.global_step >= state.max_steps):
                result = evaluate_asr(model, processor, read(data / 'val_asr_clips.json'), torch, np, run, state.global_step)
                val_history.append({'step': state.global_step, 'wer_percent': result['wer_percent'], 'checkpoint': str(run / 'checkpoints' / f'checkpoint-{state.global_step}')})
                best = min(val_history, key=lambda x: x['wer_percent'])
                if best['step'] == state.global_step:
                    model.save_pretrained(run / 'best_adapter', safe_serialization=True)
                    processor.save_pretrained(run / 'processor')
                atomic(run / 'VAL_HISTORY.json', val_history)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for k in ('loss', 'grad_norm', 'eval_loss'):
                if logs and k in logs and (not math.isfinite(float(logs[k]))):
                    raise FloatingPointError('Nonfinite training diagnostic ' + k)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows, data_collator=collator, callbacks=[Progress()])
    trainer.model_accepts_loss_kwargs = False
    atomic(run / 'RUN_MANIFEST.json', {'status': 'running', 'configuration': common, 'trainable_parameters': params})
    print('TRAIN_START', kind, 'smoke', smoke, 'examples', len(train_rows), 'trainable', params, 'GPU', torch.cuda.get_device_name(0), flush=True)
    try:
        with nullcontext():
            result = trainer.train(resume_from_checkpoint=checkpoint)
        final = run / 'adapter_final'
        model.save_pretrained(final, safe_serialization=True)
        processor.save_pretrained(run / 'processor')
        tensors = load_file(str(final / 'adapter_model.safetensors'))
        assert all((torch.isfinite(t).all().item() for t in tensors.values()))
        changed = any((not torch.equal(initial_b[n], p.detach().cpu()) for n, p in model.named_parameters() if n in initial_b))
        if checkpoint and (not changed):
            changed = any((torch.count_nonzero(t).item() > 0 for n, t in tensors.items() if '.lora_B.' in n))
        assert changed, 'LoRA matrices did not change'
        from peft import get_peft_model_state_dict
        current = get_peft_model_state_dict(model)
        assert set(current) == set(tensors), 'Saved adapter key mismatch'
        assert all((torch.equal(current[n].detach().cpu(), t) for n, t in tensors.items())), 'Saved weight readback mismatch'
        json_new(run / 'WEIGHTS_VERIFIED.json', {'adapter_sha256': sha(final / 'adapter_model.safetensors'), 'finite': True, 'changed': True, 'exact_readback': True, 'tensor_count': len(tensors)})
        val_loss = None
        atomic(run / 'RUN_MANIFEST.json', {'status': 'completed', 'configuration': common, 'trainable_parameters': params, 'step': trainer.state.global_step, 'total_steps': trainer.state.max_steps, 'adapter': str(final), 'adapter_changed': changed, 'adapter_finite': True, 'training_metrics': result.metrics, 'val_loss': val_loss, 'asr_val_history': val_history, 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
        print('TRAIN_COMPLETED', kind, smoke, final, flush=True)
    except BaseException as e:
        atomic(run / 'RUN_MANIFEST.json', {'status': 'failed_or_interrupted', 'configuration': common, 'error': repr(e), 'trainable_parameters': params, 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
        raise
