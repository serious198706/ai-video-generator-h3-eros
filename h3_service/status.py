"""Public timing contract; detailed diagnostics stay in per-attempt reports."""


def compact_timing(metrics):
    def duration(name):
        return float((metrics.get(name) or {}).get('wall_s', 0))
    values = {'conditioning_s': duration('condition'), 'generation_s': duration('generate'),
              'upscale_s': duration('upscale'),
              'gpu_wait_s': sum(float((v or {}).get('gpu_wait_s', 0)) for v in metrics.values())}
    values['processing_s'] = sum(values[k] for k in ('conditioning_s', 'generation_s', 'upscale_s'))
    return {k: round(v, 3) for k, v in values.items()}


def final_result(result, state):
    """Recovered MP4 provenance belongs to its original producer, not this retry."""
    return {'result': {k: v for k, v in result.items() if k != 'producer'},
            'producer': result.get('producer', state.get('producer'))}
