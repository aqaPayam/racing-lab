"""OpenGL 3.3 presentation, with depth testing, sun shadows and procedural materials.

GPU rendering targets an offscreen framebuffer; Pygame supplies input and HUD.
Nothing here advances physics or consumes the simulator's random generator.
"""

import numpy as np
import pygame
import moderngl
from .scene3d import Scene3D
from .car_mesh import CarMesh


VERTEX = """#version 330
in vec3 position;
in vec3 normal;
in vec3 color;
uniform mat4 vp;
uniform mat4 light_vp;
out vec3 world;
out vec3 n;
out vec3 albedo;
out vec4 shadow_pos;
void main() {
    world = position; n = normal; albedo = color;
    shadow_pos = light_vp * vec4(position, 1);
    gl_Position = vp * vec4(position, 1);
}
"""
FRAGMENT = """#version 330
in vec3 world;
in vec3 n;
in vec3 albedo;
in vec4 shadow_pos;
uniform vec3 eye;
uniform float opacity;
uniform sampler2DShadow shadow_map;
out vec4 frag;
uniform sampler2D surface_noise;
void main() {
    vec3 view = normalize(eye-world);
    vec3 norm = normalize(n);
    if(dot(norm,view)<0.0) norm=-norm;
    vec3 sun = normalize(vec3(-0.6,-0.4,1));
    float diffuse = max(dot(norm,sun),0.0);
    vec3 q = shadow_pos.xyz/shadow_pos.w*0.5+0.5;
    float visibility = 1.0;
    if(all(greaterThan(q,vec3(0))) && all(lessThan(q,vec3(1)))) {
        visibility=0.0;
        float bias=max(0.0012,0.0025*(1.0-diffuse));
        for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++)
            visibility+=texture(shadow_map,vec3(q.xy+vec2(x,y)/2048.0,q.z-bias))/9.0;
    }
    vec3 base = albedo;
    if((abs(base.r-base.g)<.06 && abs(base.g-base.b)<.08) || base.g>base.r*1.15) {
        float detail = texture(surface_noise,world.xy*.3).r;
        float fade = 1.0-smoothstep(12.0,75.0,length(eye-world));
        base *= 0.96 + 0.08*texture(surface_noise,world.xy*.03).g + .06*(detail-.5)*fade;
        if(base.g>base.r*1.15) base*=.90+.2*texture(surface_noise,world.xy*.008).b;
    }
    bool paint = base.r>base.g*2.0 && world.z>.12;
    float spec = pow(max(dot(reflect(-sun,norm),view),0.0),paint?70.0:28.0);
    vec3 lit=base*(.52+.58*diffuse*visibility)+vec3(spec*visibility*(paint?.45:.055));
    float fog=1.0-exp(-length(world-eye)*.0013);
    lit=mix(lit,vec3(.65,.75,.79),fog);
    frag=vec4(lit,opacity);
}
"""
SHADOW_VERTEX = """#version 330
in vec3 position;
uniform mat4 vp;
void main(){gl_Position=vp*vec4(position,1);}
"""
SKY_VERTEX = """#version 330
out vec2 uv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.0-1.0,0,1);}
"""
SKY_FRAGMENT = """#version 330
in vec2 uv;
out vec4 frag;
void main(){frag=vec4(mix(vec3(.77,.84,.85),vec3(.25,.48,.68),clamp(uv.y,0,1)),1);}
"""


def look_at(eye, target):
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 0, 1], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = [right, up, -forward]
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def perspective(fov, aspect, near=0.15, far=1800):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )


def matrix_bytes(matrix):
    return np.asarray(matrix.T, dtype="f4").tobytes()


def vertex_data(mesh, colors):
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-10)
    return np.concatenate(
        [mesh, np.repeat(normals[:, None], 3, axis=1), np.repeat(colors[:, None] / 255, 3, axis=1)],
        axis=2,
    ).astype("f4")


class GPUScene(Scene3D):
    def __init__(self):
        super().__init__()
        self.defer_terrain = True
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception:
            # EGL supports headless Linux research hosts without an X server.
            self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        ctx = self.ctx
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.shadow_program = ctx.program(
            vertex_shader=SHADOW_VERTEX, fragment_shader="#version 330\nvoid main(){}"
        )
        self.sky_program = ctx.program(vertex_shader=SKY_VERTEX, fragment_shader=SKY_FRAGMENT)
        self.sky = ctx.vertex_array(self.sky_program, [])
        self.depth = ctx.depth_texture((2048, 2048))
        self.depth.repeat_x = self.depth.repeat_y = False
        self.shadow = ctx.framebuffer(depth_attachment=self.depth)
        self.target = ctx.simple_framebuffer((1200, 900), components=4, samples=4)
        self.resolved = ctx.simple_framebuffer((1200, 900), components=4)
        rng = np.random.default_rng(27)
        pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.surface_noise = ctx.texture((256, 256), 3, pixels.tobytes())
        self.surface_noise.build_mipmaps()
        self.car = CarMesh()
        self.static = None
        self.dynamic = None
        self.ghost_mesh = None
        self.renderer_name = ctx.info["GL_RENDERER"]

    def upload(self, mesh, colors):
        buf = self.ctx.buffer(vertex_data(mesh, colors).tobytes())
        vao = self.ctx.vertex_array(
            self.program, [(buf, "3f 3f 3f", "position", "normal", "color")]
        )
        shadow = self.ctx.vertex_array(self.shadow_program, [(buf, "3f 24x", "position")])
        return buf, vao, shadow

    @staticmethod
    def release_mesh(mesh):
        if mesh:
            for item in reversed(mesh):
                item.release()

    def build_track(self, track):
        super().build_track(track)
        from .scenery import details

        # Replace the fallback's cone foliage with the GPU scene's denser canopy.
        start, end = self.tree_face_range
        del self.faces[start:end]
        self.faces += details(track)
        from .terrain import warp_mesh, ground_mesh

        mesh, colors = self.pack(self.faces)
        mesh = warp_mesh(track, mesh)
        ground = ground_mesh(track)
        mesh = np.concatenate([mesh, ground])
        colors = np.concatenate([colors, np.tile([81, 112, 61], (len(ground), 1))])
        self.release_mesh(self.static)
        self.static = self.upload(mesh, colors)

    def draw(self, viewer):
        env = viewer.env
        if self.track_hash != env.track.digest:
            self.build_track(env.track)
        ctx = self.ctx
        mesh, colors = self.car.mesh(env)
        data = vertex_data(mesh, colors).tobytes()
        if self.dynamic is None:
            self.dynamic = self.upload(mesh, colors)
        else:
            self.dynamic[0].write(data)
        from .terrain import pose_frame

        pos, frame = pose_frame(env)
        forward = frame[0]
        if self.camera == 0:
            eye = pos - forward * 8.7 + [0, 0, 2.9]
            target = pos + forward * 17 + [0, 0, 0.65]
            fov = 62.0
        elif self.camera == 1:
            eye = pos + forward * 0.65 + [0, 0, 1.12]
            target = pos + forward * 40 + [0, 0, 1.05]
            fov = 72.0
            if viewer.camera_motion:
                eye += frame[2] * env.vehicle.pitch * 0.3 + frame[1] * env.vehicle.roll * 0.2
        else:
            eye = pos - forward * 22 + [0, 0, 48]
            target = pos + forward * 10
            fov = 55.0
        vp = perspective(fov, 1200 / 900) @ look_at(eye, target)
        light_eye = pos + np.array([-90, -60, 150])
        ortho = np.diag([1 / 95, 1 / 95, -2 / 350, 1.0])
        ortho[2, 3] = -1
        light_vp = ortho @ look_at(light_eye, pos)
        self.shadow.use()
        self.shadow.clear(depth=1)
        ctx.enable(moderngl.DEPTH_TEST)
        self.shadow_program["vp"].write(matrix_bytes(light_vp))
        self.static[2].render()
        self.dynamic[2].render()
        self.target.use()
        self.target.clear(0.65, 0.76, 0.82, depth=1)
        ctx.disable(moderngl.DEPTH_TEST)
        self.sky.render(vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
        self.program["vp"].write(matrix_bytes(vp))
        self.program["light_vp"].write(matrix_bytes(light_vp))
        self.program["eye"].value = tuple(eye)
        self.depth.use(0)
        self.program["shadow_map"].value = 0
        self.surface_noise.use(1)
        self.program["surface_noise"].value = 1
        self.program["opacity"].value = 1.0
        self.static[1].render()
        self.dynamic[1].render()
        if viewer.show_ghost and viewer.practice:
            pose = viewer.practice.ghost(env.steps / 60)
            if pose is not None:
                from .practice import ghost_environment

                gm, gc = self.car.mesh(ghost_environment(env.track, pose, env.steps / 60))
                gc[:] = [65, 220, 240]
                if self.ghost_mesh is None:
                    self.ghost_mesh = self.upload(gm, gc)
                else:
                    self.ghost_mesh[0].write(vertex_data(gm, gc).tobytes())
                self.program["opacity"].value = 0.28
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                ctx.depth_mask = False
                self.ghost_mesh[1].render()
                ctx.depth_mask = True
                ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(self.resolved, self.target)
        image = pygame.image.frombuffer(
            self.resolved.read(components=4, alignment=1), (1200, 900), "RGBA"
        )
        image.set_alpha(None)
        viewer.screen.blit(pygame.transform.flip(image, False, True), (0, 0))
        self.hud(viewer)

    def close(self):
        self.release_mesh(self.static)
        self.release_mesh(self.dynamic)
        self.release_mesh(self.ghost_mesh)
        for item in (
            self.target,
            self.resolved,
            self.surface_noise,
            self.shadow,
            self.depth,
            self.sky,
            self.program,
            self.shadow_program,
            self.sky_program,
        ):
            item.release()
        self.ctx.release()
