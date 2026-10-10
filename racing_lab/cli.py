import argparse
import json
from pathlib import Path
import numpy as np
from .env import RacingEnv, DT
from .track import names, Track
from .data import Recorder, load
from .learning import Policy, train, reference_driver


def play(args):
    import pygame
    env = RacingEnv(args.track, max_seconds=args.seconds)
    env.reset(seed=args.seed)
    env.render()
    clock = pygame.time.Clock()
    recorder = Recorder(env,seed=args.seed)
    action = np.zeros(3,np.float32)
    paused = False
    running = True
    pending_track = args.track
    def save(label, reason):
        path = recorder.save(args.data,label,reason)
        if path:
            print(f"Saved {path} ({label})",flush=True)
    try:
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                    save("unlabeled","user_exit")
                    running=False
                if event.type!=pygame.KEYDOWN:
                    continue
                if event.key==pygame.K_c:
                    env.viewer.zoom=not env.viewer.zoom
                if event.key==pygame.K_r and not paused:
                    paused=True
                if paused and event.key in (pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4):
                    pending_track = names()[event.key-pygame.K_1]
                if paused and event.key in (pygame.K_g,pygame.K_b,pygame.K_u):
                    save({pygame.K_g:"good",pygame.K_b:"bad",pygame.K_u:"unlabeled"}[event.key],
                         "completed" if env.info()["success"] else "interrupted_or_timeout")
                    env.track=Track(pending_track)
                    env.reset(seed=args.seed)
                    env.viewer.trail=[]
                    recorder=Recorder(env,seed=args.seed)
                    action[:]=0
                    paused=False
            if not running:
                break
            if not paused:
                keys=pygame.key.get_pressed()
                target=np.array([int(keys[pygame.K_d] or keys[pygame.K_RIGHT])-int(keys[pygame.K_a] or keys[pygame.K_LEFT]),
                                 int(keys[pygame.K_w] or keys[pygame.K_UP]),int(keys[pygame.K_s] or keys[pygame.K_DOWN])])
                action += np.clip(target-action,-np.array([2.8,3,5])*DT,np.array([2.8,3,5])*DT)
                transition=env.step(action)
                recorder.append(action,transition)
                paused=transition[2] or transition[3]
            if paused:
                env.viewer.note=f"Label: G good / B bad / U unlabeled | next: {pending_track} | 1 Monza 2 Silverstone 3 Spa 4 Suzuka"
            else:
                env.viewer.note="RECORDING | R finishes this run for labeling | C switches camera | fixed simulation: 60 Hz"
            env.render()
            clock.tick(60)
    finally:
        env.close()


def replay(args):
    import pygame
    data=load(args.file)
    meta=data["metadata"]
    track=next((n for n in names() if Track(n).digest==meta["track_sha256"]),None)
    if track is None:
        raise ValueError("Recording track hash does not match an installed layout")
    env=RacingEnv(track)
    env.reset()
    clock=pygame.time.Clock()
    try:
        for i,state in enumerate(data["states"]):
            env.pos=state[:2].copy()
            env.heading,env.speed,env.lateral_speed,env.yaw_rate,env.steering=state[2:]
            env.steps=i
            env.progress=float(data["reward_features"][:i,0].sum())
            env.offtrack_steps=int(round(-data["reward_features"][:i,2].sum()/meta["dt"]))
            gate=env.next_checkpoint*env.track.length/40
            if not env.info()["offtrack"] and gate<=env.progress<=gate+env.track.width:
                env.next_checkpoint+=1
            env.smoothness=float(-data["reward_features"][:i,3].sum())
            env.render()
            env.viewer.note=f"REPLAY | {meta['label']} | {i+1}/{len(data['states'])} | ESC exits"
            if any(e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE) for e in pygame.event.get()):
                break
            clock.tick(round(1/meta["dt"]))
    finally:
        env.close()


def evaluate(args):
    policy=Policy(args.model) if args.model else None
    rows=[]
    for track in (names() if args.track=="all" else [args.track]):
        env=RacingEnv(track,max_seconds=args.seconds)
        try:
            for episode in range(args.episodes):
                seed=args.seed+episode
                obs,_=env.reset(seed=seed,options={"lateral_jitter":args.jitter})
                total=0.
                if env.viewer:
                    env.viewer.trail=[]
                while True:
                    action=policy(obs) if policy else reference_driver(env)
                    obs,reward,terminated,truncated,info=env.step(action)
                    total+=reward
                    if args.render:
                        import pygame
                        env.render()
                        env.viewer.note="AUTONOMOUS EVALUATION | ESC exits"
                        if any(e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE) for e in pygame.event.get()):
                            return
                        pygame.time.Clock().tick(60)
                    if terminated or truncated:
                        break
                rows.append({"track_id":track,"track_sha256":env.track.digest,"seed":seed,
                             "policy":str(args.model or "reference"),"return":total,**info})
                print(json.dumps(rows[-1]),flush=True)
        finally:
            env.close()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description="Racing Lab: drive, record, imitate, evaluate")
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("play",help="Drive and record demonstrations")
    p.add_argument("--track",choices=names(),default="monza")
    p.add_argument("--data",default="data")
    p.add_argument("--seconds",type=float,default=300)
    p.add_argument("--seed",type=int,default=0)
    p.set_defaults(func=play)
    p=commands.add_parser("replay")
    p.add_argument("file")
    p.set_defaults(func=replay)
    p=commands.add_parser("train")
    p.add_argument("--data",default="data")
    p.add_argument("--output",default="models/bc.npz")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--seed",type=int,default=0)
    p.set_defaults(func=lambda a:print(json.dumps(train(a.data,a.output,a.seed,a.epochs),indent=2)))
    p=commands.add_parser("evaluate")
    p.add_argument("--model",help="NPZ behavior cloning model; omit for geometric reference controller")
    p.add_argument("--track",choices=names()+["all"],default="all")
    p.add_argument("--episodes",type=int,default=3)
    p.add_argument("--seconds",type=float,default=300)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--jitter",type=float,default=1)
    p.add_argument("--render",action="store_true")
    p.add_argument("--output",default="results/evaluation.json")
    p.set_defaults(func=evaluate)
    p=commands.add_parser("inspect")
    p.add_argument("file")
    p.set_defaults(func=lambda a:print(json.dumps(load(a.file)["metadata"],indent=2)))
    args=parser.parse_args()
    try:
        args.func(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
