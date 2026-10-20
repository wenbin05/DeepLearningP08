"""Run in an existing Colab notebook after its P08 setup cell; not a shell CLI.
Uses the existing isolated Python, GPU, and Drive mount. Exports only real results.
"""
required = ['ROOT', 'protocol', 'LOCK', 'MODEL_REVISION', 'CNN_REVISION', 'run']
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError('Run the P08 notebook setup cells first. Missing: ' + ', '.join(missing))

# 4. Bounded pilot. Existing full-run outputs are preserved in their original folder.
PILOT_MODES = globals().get('PILOT_MODES', ['zero_shot', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn'])
# If time permits, rerun this cell with the extra baselines appended:
# PILOT_MODES += ['all_attn', 'full_ft', 'prefix']
PILOT = ROOT.parent / 'pilot_seed42_train1024_val64_steps32_v1'
PILOT.mkdir(parents=True, exist_ok=True)
pilot_protocol = dict(protocol, train_samples=1024, validation_samples=64,
                      max_steps=32, epochs=1, warmup_steps=4, training_seed=42,
                      purpose='single-seed short-training validation pilot')
pilot_protocol_path = PILOT / 'pilot_protocol.json'
if pilot_protocol_path.exists() and json.loads(pilot_protocol_path.read_text()) != pilot_protocol:
    raise RuntimeError('Existing pilot protocol differs; use a new PILOT directory.')
pilot_protocol_path.write_text(json.dumps(pilot_protocol, indent=2) + '\n')
(PILOT / 'environment-lock.txt').write_text(LOCK.read_text())
for mode in PILOT_MODES:
    directory = PILOT / f'{mode}_r8_q_proj,v_proj_seed42'
    completed = directory / 'validation_results.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        manifest = saved['manifest']
        config = manifest['config']
        val = manifest['data']['validation']
        train = manifest['data']['train']
        expected = dict(placement=mode, model_revision=MODEL_REVISION,
                        dataset_revision=CNN_REVISION, lora_rank=8,
                        projections='q_proj,v_proj', fp16=True, warmup_steps=4,
                        num_train_epochs=1, data_seed=42)
        if saved.get('status') != 'completed' or manifest['seed'] != 42 or any(config[k] != v for k,v in expected.items()) or val['samples'] != 64 or manifest['training_arguments']['max_steps'] != 32 or (mode != 'zero_shot' and train['samples'] != 1024):
            raise RuntimeError(f'Unexpected saved pilot protocol: {directory}')
        print('Completed previously:', mode)
    elif directory.exists():
        raise RuntimeError(f'Incomplete run preserved: {directory}. Move it outside the pilot folder before retrying this mode.')
    else:
        run(['-m', 'src.train', '--placement', mode, '--seed', '42', '--fp16',
             '--max_train_samples', '1024', '--max_eval_samples', '64',
             '--max_steps', '32', '--num_train_epochs', '1', '--warmup_steps', '4',
             '--model_revision', MODEL_REVISION, '--dataset_revision', CNN_REVISION,
             '--output_dir', PILOT], f'pilot_{mode}_seed42.log')
print('Pilot files:', PILOT)


# 5. Generate a report table from completed genuine outputs, even if only some runs finished.
# This table reports VALIDATION SUBSET scores, not full CNN/DailyMail test scores.
from IPython.display import Markdown, display
rows = []
records = []
for path in sorted(PILOT.glob('*/validation_results.json')):
    result = json.loads(path.read_text())
    if result.get('status') != 'completed':
        continue
    m = result['manifest']
    scores = result['metrics']
    if m['seed'] != 42 or m['data']['validation']['samples'] != 64 or m['training_arguments']['max_steps'] != 32:
        raise RuntimeError(f'Not a matching pilot result: {path}')
    if m['config']['placement'] != 'zero_shot' and m['data']['train']['samples'] != 1024:
        raise RuntimeError(f'Different training subset size: {path}')
    rows.append('| {} | {:,} | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.1f} |'.format(
        m['config']['placement'], m['parameters']['trainable'],
        scores['validation_rouge1'], scores['validation_rouge2'], scores['validation_rougeL'],
        scores['validation_loss'], result['training_seconds']))
    records.append(dict(source_file=str(path), result=result))
if not rows:
    raise RuntimeError('No completed pilot outputs yet. Do not invent missing entries.')
caption = ('Single-seed pilot (seed 42): fixed 1,024-example CNN/DailyMail training subset; '
           '32 optimizer updates for trained models; 64-example validation subset; '
           'official stemmed ROUGE F1 ×100 against full references. No standard deviations '
           'are available. Short training is not evidence of convergence or published-baseline reproduction.')
table = ('| Configuration | Trainable parameters | ROUGE-1 | ROUGE-2 | ROUGE-L | Validation loss | Training seconds |\n'
         '|---|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n' + caption)
(PILOT / 'preliminary_table.md').write_text(table + '\n')
(PILOT / 'completed_pilot_outputs.json').write_text(json.dumps(records, indent=2) + '\n')
display(Markdown(table))
print('For report revision, share preliminary_table.md and completed_pilot_outputs.json from:', PILOT)
