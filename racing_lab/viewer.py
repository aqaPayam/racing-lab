"""Pygame presentation only: never changes simulator state."""
import numpy as np
import pygame
from .vehicle import CAR_LENGTH, CAR_WIDTH, WHEELBASE


class Viewer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((1200, 900))
        pygame.display.set_caption("Racing Lab | human demonstrations & machine learning")
        self.font = pygame.font.SysFont("consolas", 19)
        self.title = pygame.font.SysFont("consolas", 30, bold=True)
        self.note = ""
        self.zoom = True
        self.trail = []

    def draw(self, rgb=False):
        env, screen = self.env, self.screen
        pts = env.track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        scale = min(850/(hi-lo)[0], 700/(hi-lo)[1])
        center = (hi+lo)/2
        if self.zoom:
            scale = max(4., 8.-env.speed*.07)
            center = env.pos + min(40.,env.speed*.7)*np.array([np.cos(env.heading),np.sin(env.heading)])
        def xy(p):
            return tuple(np.rint((p-center)*scale+np.array([465, 465])).astype(int))
        screen.fill((14, 23, 31))
        pygame.draw.rect(screen, (23, 49, 42), (20, 90, 900, 770), border_radius=20)
        screen.set_clip(pygame.Rect(20,90,900,770))
        road = [xy(p) for p in pts]
        width = max(5, int(env.track.width*scale))
        # Segment rectangles in draw.lines leave gaps at diagonal joins; circles fill them.
        for color,w in [((194,67,62),width+7),((184,194,187),width+3),((58,69,78),width)]:
            pygame.draw.lines(screen,color,True,road,w)
            for p in road:
                pygame.draw.circle(screen,color,p,w//2)
        for i in range(0, len(road), 10):
            pygame.draw.line(screen, (110, 122, 123), road[i], road[(i+3)%len(road)], 1)
        start = env.track.points[0]
        normal = np.array([-env.track.tangents[0,1], env.track.tangents[0,0]])
        pygame.draw.line(screen, (245,245,240), xy(start-normal*env.track.width/2), xy(start+normal*env.track.width/2), 4)
        self.trail.append(env.pos.copy())
        self.trail = self.trail[-1800:]
        if len(self.trail)>1:
            pygame.draw.lines(screen, (57, 169, 151), False, [xy(p) for p in self.trail], 2)
        h = env.heading
        forward = np.array([np.cos(h),np.sin(h)])
        right = np.array([-forward[1],forward[0]])
        def carpoint(x,y):
            return xy(env.pos+x*forward+y*right)
        for x in [-WHEELBASE/2, WHEELBASE/2]:
            for y in [-CAR_WIDTH/2+.2, CAR_WIDTH/2-.2]:
                pygame.draw.polygon(screen,(9,12,18),[carpoint(x+dx,y+dy) for dx,dy in [(-.36,-.2),(.36,-.2),(.36,.2),(-.36,.2)]])
        half=CAR_LENGTH/2
        pygame.draw.polygon(screen, (255, 195, 65), [carpoint(half,0),carpoint(1,.55),carpoint(-half,.6),carpoint(-half,-.6),carpoint(1,-.55)])
        pygame.draw.circle(screen, (20,30,40), carpoint(-.5,0), max(1,round(.25*scale)))
        if not self.zoom:
            pygame.draw.circle(screen,(255,195,65),xy(env.pos),8,1)
        pygame.draw.line(screen,(230,240,240),(45,825),(45+round(10*scale),825),2)
        screen.blit(self.font.render("10 m",True,(230,240,240)),(45,833))
        screen.set_clip(None)
        screen.blit(self.title.render("RACING LAB", True, (233,240,245)), (30,25))
        screen.blit(self.font.render(env.track.name, True, (89,210,181)), (300,34))
        info = env.info()
        lines = ["TELEMETRY", "", f"{env.speed*3.6:6.1f} km/h", f"TIME  {info['elapsed']:6.2f}s",
                 f"LAP   {max(0,info['progress'])*100:6.1f}%", f"GATES {info['checkpoint']:2d}/40", "",
                 "OFF TRACK" if info['offtrack'] else "ON TRACK", "", "W / UP    throttle", "S / DOWN  brake",
                 "A D / <>  steering", "C         camera", "R         new run", "ESC       save/exit", "",
                 "After each run:", "G = good", "B = bad", "U = unlabeled", "", "1-4 change track", "(between runs)"]
        for i,line in enumerate(lines):
            screen.blit(self.font.render(line, True, (216,225,230) if i!=7 else (255,184,80)), (945,110+i*27))
        screen.blit(self.font.render(self.note[:105], True, (255,211,110)), (30,868))
        pygame.display.flip()
        if rgb:
            return np.transpose(pygame.surfarray.array3d(screen), (1,0,2))

    def close(self):
        pygame.quit()
