import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from racing_lab import RacingEnv
from racing_lab.track import names
from racing_lab.data import Recorder,load
from racing_lab.learning import reference_driver,train,Policy


def test_gym_contract():
    env=RacingEnv()
    check_env(env,skip_render_check=True)


def test_determinism_and_invalid_action():
    a,b=RacingEnv(),RacingEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    for _ in range(120):
        action=[.1,.6,0]
        assert np.array_equal(a.step(action)[0],b.step(action)[0])
    with pytest.raises(ValueError):
        a.step([np.nan,1,0])


@pytest.mark.parametrize("track",names())
def test_reference_completes_all_gates(track):
    env=RacingEnv(track,max_seconds=300)
    env.reset()
    for _ in range(env.max_steps):
        _,_,done,truncated,info=env.step(reference_driver(env))
        if done or truncated:
            break
    assert info["success"], info
    assert info["checkpoint"]>=40
    assert info["offtrack_fraction"]<.05


def test_timeout_is_not_success():
    env=RacingEnv(max_seconds=.05)
    env.reset()
    for _ in range(3):
        _,_,terminated,truncated,info=env.step([0,0,0])
    assert truncated and not terminated
    assert info["lap_time"] is None
    with pytest.raises(RuntimeError):
        env.step([0,0,0])


def test_skipped_gate_cannot_complete_lap():
    env=RacingEnv()
    env.reset()
    env.progress=env.track.length+1
    assert not env.info()["success"]


def test_dataset_and_training_roundtrip(tmp_path):
    for seed in range(3):
        env=RacingEnv()
        env.reset(seed=seed)
        rec=Recorder(env,seed=seed)
        initial=env.observe().copy()
        for _ in range(90):
            action=reference_driver(env)
            rec.append(action,env.step(action))
        path=rec.save(tmp_path,"good")
        data=load(path)
        assert data["observations"].shape==(91,20)
        assert data["actions"].shape==(90,3)
        assert np.array_equal(data["observations"][0],initial)
        assert np.allclose(data["states"][-1],env.state())
        restored=RacingEnv()
        restored.reset(seed=seed)
        for action in data["actions"]:
            restored.step(action)
        assert np.array_equal(restored.state(),data["states"][-1])
    output=tmp_path/"model.npz"
    report=train(tmp_path,output,epochs=3)
    assert not set(report["train_runs"]) & set(report["validation_runs"])
    action=Policy(output)(initial)
    assert env.action_space.contains(action)


def test_rgb_render():
    env=RacingEnv(render_mode="rgb_array")
    env.reset()
    pixels=env.render()
    assert pixels.shape==(900,1200,3)
    assert pixels.std()>10
    env.close()


def test_play_save_label_and_exit(tmp_path,monkeypatch):
    import argparse
    import pygame
    from racing_lab.cli import play,replay
    frames=iter(range(100))
    def events():
        frame=next(frames)
        key={15:pygame.K_r,16:pygame.K_g,30:pygame.K_ESCAPE}.get(frame)
        return [pygame.event.Event(pygame.KEYDOWN,key=key)] if key else []
    class Keys:
        def __getitem__(self,key):
            return key==pygame.K_w
    monkeypatch.setattr(pygame.event,"get",events)
    monkeypatch.setattr(pygame.key,"get_pressed",lambda:Keys())
    play(argparse.Namespace(track="monza",seconds=300,seed=0,data=tmp_path))
    files=list(tmp_path.glob("*.npz"))
    assert len(files)==2
    datasets=[load(p) for p in files]
    assert {d["metadata"]["label"] for d in datasets}=={"good","unlabeled"}
    assert all(d["states"][-1,3]>0 for d in datasets)
    monkeypatch.setattr(pygame.event,"get",lambda:[pygame.event.Event(pygame.QUIT)])
    replay(argparse.Namespace(file=files[0]))
