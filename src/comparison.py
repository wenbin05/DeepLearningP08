"""Validate comparisons across configurations, not only seeds within a run group."""
PLACEMENTS = {'encoder_self_attn', 'decoder_self_attn', 'cross_attn'}


def validate_comparison(rows):
    if not rows:
        raise ValueError('No results to compare')

    def evaluation(row):
        p = row['protocol']
        return (p['evaluation'] or p['data']['validation'], p['generation'],
                p['metric_protocol'], p['reference_protocol'], p.get('metric_revision'))

    if any(evaluation(r) != evaluation(rows[0]) for r in rows):
        raise ValueError('Incompatible data or generation protocols in comparison')

    # Placement, rank, projection and prefix-length are explicit experimental axes.
    # Everything else, including preprocessing, base revision and precision, is shared.
    def shared(row):
        p = row['protocol']
        config = {k: v for k, v in p['config'].items()
                  if k not in ('placement', 'lora_rank', 'projections', 'prefix_tokens')}
        return (config, p.get('model_config'), p.get('environment'),
                p.get('evaluation_environment'), p.get('checkpoint_selection'), row['seeds'])

    if any(shared(r) != shared(rows[0]) for r in rows):
        raise ValueError('Incompatible model, environment, seeds or configuration in comparison')

    trained = [r for r in rows if r['placement'] != 'zero_shot']
    def exposure(row):
        p = row['protocol']
        args = dict(p['training_arguments'])
        # Full FT deliberately has a different LR; verify it against its declared config.
        config = p['config']
        lr_key = 'lr_full_ft' if row['placement'] == 'full_ft' else 'learning_rate'
        if lr_key in config and args.get('learning_rate') != config[lr_key]:
            raise ValueError('Training learning rate differs from declared configuration')
        args.pop('learning_rate', None)
        return (p['data'], args)

    if trained:
        reference = exposure(trained[0])
        if any(exposure(r) != reference for r in trained):
            raise ValueError('Incompatible training conditions in comparison')

    budgets = {}
    for row in trained:
        if row['placement'] in PLACEMENTS:
            key = (row['rank'], row['projections'])
            count = row['parameters']['trainable']
            if key in budgets and count != budgets[key]:
                raise ValueError('Isolated placements have unequal trainable parameter budgets')
            budgets[key] = count
