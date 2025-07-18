#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

TEMPLATE = '''import bpy, mathutils, math

# ───────────────────────────────────────────
# 0. シーンをクリア
# ───────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ───────────────────────────────────────────
# 1. 入力データ
# ───────────────────────────────────────────
coords = {{
{coords_entries}
}}

vertex_edge_route = {vertex_edge_route}

eid2info = {{
{eid2info_entries}
}}

flat_vertex = {flat_vertex}

z_flags = {{
{z_flags_entries}
}}

preferred_signs = {{
{pref_entries}
}}

# ───────────────────────────────────────────
# 2. flat_vertex に球を追加
# ───────────────────────────────────────────
fillet_r = 0.5
sphere_r = fillet_r * 0.1
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=sphere_r,
    location=(
        coords[str(flat_vertex)][0],
        coords[str(flat_vertex)][1],
        0.0
    ),
    segments=32,
    ring_count=16
)

# ───────────────────────────────────────────
# 3. 制御点リスト構築 (頂点→辺→頂点→…)
# ───────────────────────────────────────────
r       = fillet_r
off_mag = 0.1

delta_z = 0.2  # 交差時のZオフセットを増量

pts      = []
pt_types = []  # 各制御点の種類: 'P', 'M'
route    = vertex_edge_route
L        = len(route)
edge_counter = {{}}
# 通過カウンタ: 同じ頂点の2回通過で上下を交互に

pts      = []
pt_types = []
L = len(route)

for i, key in enumerate(route):
    if i % 2 == 0:
        # 頂点
        v = str(key)
        x, y = coords[v]
        if v == str(flat_vertex):  # flat_vertexならz=0
            z = 0.0
        else:
            eid = str(route[i-1]) if i > 0 else str(route[-2])
            z_flag_key = f"{{eid}}-{{v}}"
            sign = z_flags.get(z_flag_key, 1)
            z = sign * delta_z
        P = mathutils.Vector((x, y, z))
        pts.append(P)
        pt_types.append('P')
    else:
        # 辺制御点（ここは従来通り）
        eid = str(key)
        u, v = eid2info[eid]
        mu, mv = coords[str(u)], coords[str(v)]
        Mx, My = (mu[0] + mv[0]) * 0.5, (mu[1] + mv[1]) * 0.5
        M = mathutils.Vector((Mx, My, 0.0))
        sign = - preferred_signs.get(eid, 0)
        dx, dy = mv[0] - mu[0], mv[1] - mu[1]
        perp = mathutils.Vector((-dy, dx, 0.0)).normalized()
        M += perp * off_mag * sign
        pts.append(M); pt_types.append('M')

# ───────────────────────────────────────────
# 3.5 デバッグ: 全ての制御点に小さな球を配置 & 色分け
# ───────────────────────────────────────────
mat_debug_inter = bpy.data.materials.new(name="DebugRed")
mat_debug_inter.diffuse_color = (1.0, 0.0, 0.0, 1.0)
mat_debug_edge  = bpy.data.materials.new(name="DebugBlue")
mat_debug_edge.diffuse_color  = (0.0, 0.0, 1.0, 1.0)

debug_r = sphere_r * 0.5
for p, kind in zip(pts, pt_types):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=debug_r,
        location=(p.x, p.y, p.z),
        segments=16,
        ring_count=8
    )
    obj = bpy.context.active_object
    if kind == 'P':
        obj.data.materials.append(mat_debug_inter)
    else:
        obj.data.materials.append(mat_debug_edge)

# ───────────────────────────────────────────
# 4. Bezier Curve 作成 & チューブ化
# ───────────────────────────────────────────
curve_data = bpy.data.curves.new('GraphTube', 'CURVE')
curve_data.dimensions = '3D'
spline = curve_data.splines.new('BEZIER')
spline.bezier_points.add(len(pts) - 1)
for idx, p in enumerate(pts):
    bp = spline.bezier_points[idx]
    bp.co = p
    bp.handle_left_type = bp.handle_right_type = 'AUTO'
spline.use_cyclic_u = True

curve_data.resolution_u     = 16
curve_data.bevel_depth      = 0.01
curve_data.bevel_resolution = 32

tube = bpy.data.objects.new('GraphTube', curve_data)
bpy.context.collection.objects.link(tube)

# ───────────────────────────────────────────
# 5. メッシュ変換 & スムーズシェーディング
# ───────────────────────────────────────────
bpy.context.view_layer.objects.active = tube
tube.select_set(True)
bpy.ops.object.convert(target='MESH')
tube = bpy.context.active_object
mesh = tube.data
for poly in mesh.polygons:
    poly.use_smooth = True

# ───────────────────────────────────────────
# 6. 頂点 & 辺ラベルを色違いで配置
# ───────────────────────────────────────────
mat_v = bpy.data.materials.new(name="LabelRed")
mat_v.diffuse_color = (1.0, 0.0, 0.0, 1.0)
for v, coord in coords.items():
    txt_curve = bpy.data.curves.new(name=f"V{{v}}", type='FONT')
    txt_curve.body = str(v)
    txt_obj = bpy.data.objects.new(name=f"V{{v}}", object_data=txt_curve)
    txt_obj.location = (coord[0], coord[1], sphere_r + 0.1)
    txt_obj.scale    = (0.2, 0.2, 0.2)
    txt_obj.data.materials.append(mat_v)
    bpy.context.collection.objects.link(txt_obj)

mat_e = bpy.data.materials.new(name="LabelBlue")
mat_e.diffuse_color = (0.0, 0.0, 1.0, 1.0)
edge_label_offset = 0.08  # ラベル浮かせ量（調整可能）

edge_idx = 0
for i, key in enumerate(vertex_edge_route):
    if i % 2 == 1:
        eid = key
        # M制御点のインデックスはi
        Mx, My, Mz = pts[i].x, pts[i].y, pts[i].z
        txt_curve = bpy.data.curves.new(name=f"E{{eid}}", type='FONT')
        txt_curve.body = str(eid)
        txt_obj = bpy.data.objects.new(name=f"E{{eid}}", object_data=txt_curve)
        # そのままM制御点の位置＋浮かせ
        txt_obj.location = (Mx, My, Mz + edge_label_offset)
        txt_obj.scale    = (0.2, 0.2, 0.2)
        txt_obj.data.materials.append(mat_e)
        bpy.context.collection.objects.link(txt_obj)


# ───────────────────────────────────────────
# 7. カメラ & 三灯ライティング
# ───────────────────────────────────────────
cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam'))
cam.location      = (0.0, 0.0, 10.0)
cam.rotation_euler = (math.radians(90), 0.0, 0.0)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
light = bpy.data.lights.new('L','AREA')
light.energy = 800
light_obj = bpy.data.objects.new('Light', light)
light_obj.location = (5, -5, 8)
bpy.context.collection.objects.link(light_obj)

bpy.context.scene.render.engine   = 'CYCLES'
bpy.context.scene.cycles.samples = 128

print("完了：F12でレンダリングしてください。")
'''

def make_entries(d):
    lines = []
    for k, v in d.items():
        # キーを必ずstr型でダブルクォート
        lines.append(f'    "{str(k)}": {v},')
    return "\n".join(lines)
def main():
    data = json.load(open(sys.argv[1], encoding='utf-8')) if len(sys.argv)>1 else json.load(sys.stdin)

    coords_entries     = make_entries(data['coords'])
    route_list         = data.get('vertex_edge_route', data.get('route', []))
    eid2info_entries   = make_entries(data['eid2verts'])
    flat_vertex        = f'"{data["flat_vertex"]}"' if isinstance(data['flat_vertex'], str) else data['flat_vertex']
    z_flags_entries    = make_entries(data['z_flags'])
    pref_entries       = make_entries(data['preferred_signs'])


    script = TEMPLATE.format(
        coords_entries     = coords_entries,
        vertex_edge_route  = route_list,
        eid2info_entries   = eid2info_entries,
        flat_vertex        = flat_vertex,
        z_flags_entries    = z_flags_entries,
        pref_entries       = pref_entries
    )

    with open('blender_script.py','w',encoding='utf-8') as f:
        f.write(script)
    print("生成: blender_script.py")

if __name__ == '__main__':
    main()
