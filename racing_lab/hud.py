"""Shared, presentation-only instruments for both game views."""

import pygame
import numpy as np
import math


def draw(viewer, camera):
    env, screen = viewer.env, viewer.screen
    info = env.info()
    state = env.vehicle

    def panel(rect):
        layer = pygame.Surface(rect[2:], pygame.SRCALPHA)
        layer.fill((10, 17, 25, 232))
        screen.blit(layer, rect[:2])

    def text(value, pos, color=(222, 232, 238), large=False):
        screen.blit((viewer.title if large else viewer.font).render(str(value), True, color), pos)

    panel((18, 16, 1164, 78))
    pygame.draw.rect(screen, (230, 57, 69), (18, 16, 5, 78))
    text("RACING LAB", (38, 27), large=True)
    text(env.track.name, (298, 32), (114, 218, 198))
    text(f"{camera}   C camera   V view   H recover   X reverse", (38, 67))
    text(f"{info['elapsed']:6.2f}s   LAP {max(0, info['progress']) * 100:5.1f}%", (842, 33))
    panel((18, 742, 1164, 142))
    text(f"{abs(env.speed) * 3.6:3.0f}", (38, 756), large=True)
    text("km/h", (105, 764), (151, 170, 185))
    gear = "R" if state.gear < 0 else str(state.gear)
    text(gear, (191, 756), (255, 208, 96), large=True)
    text("GEAR", (182, 800), (151, 170, 185))
    for i in range(14):
        color = (76, 206, 158) if i < 8 else ((244, 193, 69) if i < 11 else (243, 73, 86))
        if state.rpm < 4500 + (i + 1) * 700:
            color = (42, 51, 62)
        pygame.draw.rect(screen, color, (275 + i * 16, 759, 12, 12), border_radius=3)
    text(f"{state.rpm:5.0f} RPM", (275, 784))
    for i, (label, value, color) in enumerate(
        zip(("THR", "BRK"), env.action[1:], ((76, 206, 158), (238, 80, 85)))
    ):
        y = 819 + i * 24
        text(label, (38, y), (151, 170, 185))
        pygame.draw.rect(screen, (39, 50, 62), (89, y + 4, 143, 10), border_radius=4)
        pygame.draw.rect(screen, color, (89, y + 4, int(143 * value), 10), border_radius=4)
    text(
        (
            f"LONG {env.telemetry['longitudinal_g']:+.1f}g   LAT {env.telemetry['lateral_g']:+.1f}g"
            if getattr(env, "telemetry_available", True)
            else "FORCE TELEMETRY NOT RECORDED"
        ),
        (525, 756),
    )
    aids = f"{'AUTO' if env.driving.automatic else 'MANUAL'} / TC {'ON' if env.driving.traction_control else 'OFF'} / ABS {'ON' if env.driving.abs else 'OFF'}"
    text(aids, (525, 784), (114, 218, 198))
    sectors = (
        "  ".join(f"S{i + 1} {v:.2f}" for i, v in enumerate(env.sectors))
        or "SECTORS  --  /  --  /  --"
    )
    text(sectors, (525, 813))
    practice = getattr(viewer, "practice", None)
    if practice and practice.best_time is not None:
        text(f"BEST {practice.best_time:.2f}s", (915, 756), (255, 208, 96))
        delta = practice.delta(env.progress, env.steps * env.dt)
        if delta is not None:
            text(
                f"DELTA {delta:+.2f}s",
                (915, 784),
                (114, 218, 198) if delta < 0 else (244, 139, 106),
            )
    text(viewer.note[:77], (275, 849), (167, 184, 195))
    reference = getattr(viewer, "agent_ghost", None)
    if reference:
        from .agent_ghost import strict_clean

        clean = strict_clean(env)
        valid_finish = info["finished"] and clean
        panel((18, 112, 650, 140))
        text("SAC GHOST / SAME STANDING START", (38, 122), (65, 220, 208))
        text(f"SAC {reference.best_time:.2f}s   YOU {info['elapsed']:.2f}s", (38, 151))
        delta = (
            info["elapsed"] - reference.best_time
            if valid_finish
            else reference.delta(env.progress, info["elapsed"])
        )
        if viewer.race_phase == "finished" and not valid_finish:
            text("NO CLEAN LAP TIME / Space to try again", (38, 180), (255, 174, 93))
        elif delta is not None:
            status = "BEHIND" if delta > 0.005 else "AHEAD" if delta < -0.005 else "LEVEL"
            label = "FINISH GAP" if valid_finish else "GAP AT EQUAL PROGRESS"
            text(
                f"{label}: {delta:+.2f}s {status}",
                (38, 180),
                (255, 174, 93) if delta > 0 else (65, 220, 208),
            )
        status = (
            "CLEAN LAP"
            if valid_finish
            else "CLEAN SO FAR"
            if clean
            else "INVALID: off-road / collision / recovery"
        )
        text(status, (38, 211), (65, 220, 208) if clean else (255, 116, 97))
        if viewer.race_phase in ("ready", "countdown", "finished"):
            panel((340, 340, 540, 86))
            message = (
                "SPACE / ENTER TO RACE"
                if viewer.race_phase == "ready"
                else str(max(1, math.ceil(viewer.countdown)))
                if viewer.race_phase == "countdown"
                else "LAP FINISHED / SPACE AGAIN"
                if info["finished"]
                else "RUN STOPPED / SPACE AGAIN"
            )
            text(message, (365, 365), large=True)
    elif info["invalid_lap"]:
        text("RECOVERY USED / LAP INVALID", (420, 111), (255, 183, 90))
    elif env.impact > 0.4:
        text(f"BARRIER CONTACT  {env.impact * 3.6:.0f} km/h impact", (380, 111), (255, 108, 96))
    elif info["offtrack"]:
        text("ON SHOULDER", (503, 111), (255, 202, 100))
    # A compact map remains visible in both presentations.
    panel((989, 112, 193, 187))
    points = env.track.points
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = min(167 / max(1, (hi - lo)[0]), 154 / max(1, (hi - lo)[1]))
    mapped = (points - (lo + hi) / 2) * scale + [1085, 204]
    pygame.draw.lines(screen, (113, 137, 148), True, mapped, 2)
    if reference:
        if viewer.show_lines:
            pygame.draw.lines(
                screen,
                (65, 220, 208),
                False,
                (reference.states[:, :2] - (lo + hi) / 2) * scale + [1085, 204],
                1,
            )
            if viewer.human_line and len(viewer.human_line) > 1:
                pygame.draw.lines(
                    screen,
                    (255, 174, 93),
                    False,
                    (np.asarray(viewer.human_line)[:, :2] - (lo + hi) / 2) * scale + [1085, 204],
                    1,
                )
        if viewer.show_ghost:
            pose = reference.ghost(info["elapsed"])
            pygame.draw.circle(
                screen, (65, 220, 208), (pose[:2] - (lo + hi) / 2) * scale + [1085, 204], 4
            )
    pygame.draw.circle(
        screen,
        (255, 174, 93) if reference else (255, 210, 98),
        (env.pos - (lo + hi) / 2) * scale + [1085, 204],
        4,
    )
