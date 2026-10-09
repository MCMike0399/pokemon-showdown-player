import pytest
from ml.continuous import LearningConfig
from ml.resources import ResourcePolicy

def test_evaluation_priority_preserves_shared_process_budget():
    policy=ResourcePolicy(max_workers=4,evaluation_workers=3)
    assert policy.simulator_workers('collector')+policy.simulator_workers('evaluator')==4
    assert policy.simulator_workers('evaluator')==3
    with pytest.raises(ValueError):ResourcePolicy(max_workers=4,evaluation_workers=4)

def test_cpu_lane_does_not_initialize_metal_allocator(monkeypatch):
    monkeypatch.setattr('ml.resources.os.nice',lambda value:None)
    monkeypatch.setattr('torch.backends.mps.is_available',lambda:pytest.fail('CPU lane must not create a Metal context'))
    ResourcePolicy(backend='mps').apply_background(enable_mps=False)

@pytest.mark.parametrize('values',[{'learning_rate':float('nan')},{'entropy_coef':float('nan')},
 {'minibatch_size':0},{'training_epochs':31},{'target_kl':0}])
def test_invalid_training_configuration_cannot_start_jobs(values):
    with pytest.raises(ValueError):LearningConfig(**values)

def test_saved_legacy_config_retains_existing_ppo_defaults(tmp_path):
    import json
    (tmp_path/'autopilot.json').write_text(json.dumps({'enabled':True}))
    config=LearningConfig.load(tmp_path)
    assert (config.learning_rate,config.entropy_coef,config.minibatch_size,config.training_epochs,config.target_kl)==(.0003,.01,32,4,.03)
