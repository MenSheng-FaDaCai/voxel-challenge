from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges = 0, exposure=10)
scene.set_floor(-3, (0, 0, 0))
# scene.set_directional_light((0, 1, -1), 0.1, (0.996, 1, 0.765))

@ti.func
def create_block(pos, size, color, color_noise):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 1, color + color_noise * ti.random())

@ti.func
def create_leaves(pos, radius, color):
    for I in ti.grouped(
            ti.ndrange((-radius, radius), (-radius, radius),
                       (-radius, +radius))):
        f = I / radius
        h = 0.5 - max(f[1], -0.5) * 0.5
        d = vec2(f[0], f[2]).norm()
        prob = max(0, 1 - d)**2 * h  # xz mask
        prob *= h  # y mask
        # noise
        prob += ti.sin(f[0] * 5 + pos[0]) * 0.02
        prob += ti.sin(f[1] * 9 + pos[1]) * 0.01
        prob += ti.sin(f[2] * 10 + pos[2]) * 0.03
        if prob < 0.1:
            prob = 0.0
        if ti.random() < prob:
            scene.set_voxel(pos + I, 1, color + (ti.random() - 0.5) * 0.2)

@ti.func
def create_tree(pos, height, radius, color):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + 3), (pos[1], pos[1] + height - radius * 0.5),
                       (pos[2], pos[2] + 3))):
        scene.set_voxel(I, 1, vec3(0.4,0.2,0) + vec3(0.3) * ti.random())

    # Leaves
    create_leaves(pos + ivec3(0, height, 0), radius, color)

    # Ground
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        prob = max((radius - vec2(i, j).norm()) / radius, 0)
        prob = prob * prob
        if ti.random() < prob * prob:
            scene.set_voxel(pos + ivec3(i, 1, j), 1,color + ti.random() * vec3(0.1))

@ti.func
def draw_rectangle(start, width, height, deep, color, color_noise, mat=1):
    for i,j,k in ti.ndrange((start[0],start[0]+width),(start[1],start[1]+height),(start[2],start[2]+deep)):
        scene.set_voxel(vec3(i,j,k),mat,color+color_noise*ti.random())

@ti.func
def draw_logo(start, color, color_noise, mat=1,num=5):
    for i, j in ti.ndrange((start[0], start[0] + num), (start[1], start[1] + num)):
        if (i - start[0])!=2 and (j - start[1])!=2:
            scene.set_voxel(vec3(i, j, start[2]), mat, color + color_noise * ti.random())

@ti.func
def draw_screen(start, width, height, deep):
    for i,j,k in ti.ndrange((start[0],start[0]+width),(start[1],start[1]+height),(start[2],start[2]+deep)):
        color=vec3((i-start[0])/255.0,(j-start[1])/510.0,(k-start[2])/255.0+0.3)
        scene.set_voxel(vec3(i,j,k),2,color)

@ti.func
def draw_pot(start, width, height, deep):
    for i, j, k in ti.ndrange((start[0], start[0] + width), (start[1], start[1]+height), (start[2], start[2] + deep)):
        color=vec3(0.5,0.4,0.3);mat=0
        if i == start[0] or i == start[0] + width - 1 or k == start[2] or k == start[2] + deep - 1:
            color = vec3(0.1,0.1,0.2)
            if j!=start[1]+height-1:
                mat=1
        scene.set_voxel(vec3(i,j,k),mat,color)

@ti.kernel
def initialize_voxels():
    draw_screen(ivec3(0,0,1),40,30,1)
    draw_rectangle(ivec3(-1,-1,2),42,32,1,vec3(0.1, 0.1, 0.2), 0, 1)
    draw_rectangle(ivec3(-1,0,-31),42,1,34,vec3(0.1, 0.1, 0.2),0,1)
    draw_rectangle(ivec3(1,0,-16),38,1,17,vec3(0.063,0.063,0.063),0,1)
    draw_logo(ivec3(18,14,2),vec3(0.1,0.1,0.5),0,2,5)
    draw_pot(ivec3(-30,0,-30),20,5,20)
    create_tree(ivec3(-20, 0, -20), 25, 15, vec3(0, 0.6, 0.4))
    draw_rectangle(ivec3(-10,-10,-63),50,20,1,vec3(0.25,0.17,0),0,1)
    draw_rectangle(ivec3(-40,-3,-45),100,2,60,vec3(0.3,0.2,0) + vec3(0.1) * ti.random(),0,1)

initialize_voxels()
scene.finish()