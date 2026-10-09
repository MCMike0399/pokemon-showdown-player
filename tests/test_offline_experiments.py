"""Experimental controls preserve collecting-policy validation and episode accounting."""
import numpy as np
import pytest
import torch
from ml.features import SCHEMA, STATE_DIM, ACTION_DIM
from ml.model import Model
from experiments.offline_ppo import scores, Compute, paired_scout_scores

FMT = 'gen9championsvgc2026regmc'

def episode(model, number):
    state = np.zeros(STATE_DIM,dtype=np.float32)
    actions = np.zeros((2,ACTION_DIM),dtype=np.float32)
    actions[1,-1] = 1.5
    prediction = model.predict(state,actions,selected_index=number%2)
    return {'id':str(number),'format':FMT,'revision':model.revision,'schema':SCHEMA,
        'status':'complete','on_policy':True,'source':'local','outcome':1 if number%2 else -1,
        'steps':[{'state':state.tolist(),'actions':actions.tolist(),'index':prediction['index'],
                  'logprob':prediction['logprob'],'value':prediction['value'],'choice':f'move {prediction["index"]+1}'}]}

def test_training_controls_change_optimizer_and_entropy_without_accepting_stale_rollouts(tmp_path):
    models = [Model(tmp_path/str(i),FMT,seed=11) for i in range(2)]
    batches = [[episode(m,n) for n in range(12)] for m in models]
    reports = [m.train(b,epochs=1,learning_rate=.0007,entropy_coef=coef,minibatch_size=6)
               for m,b,coef in zip(models,batches,(0,.1))]
    assert all(r['trained'] and r['optimizer_steps']==2 and r['minibatch_size']==6 for r in reports)
    assert all(r['learning_rate']==.0007 and r['maximum_collecting_logprob_error']<1e-4 for r in reports)
    assert any(not torch.equal(models[0].net.state_dict()[k],models[1].net.state_dict()[k]) for k in models[0].net.state_dict())
    assert not models[0].train(batches[0])['trained']

@pytest.mark.parametrize('kwargs', [{'entropy_coef':float('nan')},{'entropy_coef':-.1},
    {'minibatch_size':0},{'minibatch_size':1.5},{'learning_rate':float('nan')}])
def test_bad_training_controls_leave_checkpoint_intact(tmp_path,kwargs):
    model = Model(tmp_path,FMT)
    before = model.path.read_bytes()
    with pytest.raises(ValueError):
        model.train([],**kwargs)
    assert before == model.path.read_bytes()

def test_unfinished_games_are_separate_from_episode_return():
    results = [{'winner':'LocalBrain'},{'winner':'LocalOpponent'},{'tie':True},{'unfinished':True}]
    report = scores(results)
    assert report['completed']==3 and report['unfinished']==1
    assert report['wins']==report['losses']==report['draws']==1
    assert report['mean_episode_return']==0 and report['win_rate']==1/3
    assert report['win_rate_wilson95'][0] < 1/3 < report['win_rate_wilson95'][1]

def test_intensive_profile_reserves_cpu_and_memory():
    with pytest.raises(ValueError):
        Compute(workers=0)
    with pytest.raises(ValueError):
        Compute(min_available_gb=0)


def test_scout_comparison_refuses_unmatched_seed_or_checkpoint():
    game = {'seed':1,'learner_side':'p1','learner_team':'a','opponent_team':'b',
            'revision':'frozen','winner':'LocalBrain'}
    rows = [{'condition':condition,'opponent':opponent,'games':[dict(game)]}
            for condition in ('without_scout','with_scout') for opponent in ('random','tactical')]
    assert all(r['unchanged']==1 and r['one_sided_exact_sign_p']==1 for r in paired_scout_scores(rows))
    rows[2]['games'][0]['revision'] = 'changed'
    with pytest.raises(ValueError,match='not paired'):
        paired_scout_scores(rows)
