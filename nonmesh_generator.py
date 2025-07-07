#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

TEMPLATE = '''import bpy, mathutils, math

# ───────────────────────────────────────────
# 0. シーンをクリア
# ───────────────────────────────────────────
#bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.delete(use_global=False)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# ───────────────────────────────────────────
# 1. 入力データ
# ───────────────────────────────────────────
coords = {{
{coords_entries}
}}

vertex_edge_routes = {vertex_edge_routes}

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

label = {label}

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
# 3. 各ルートごとに制御点リスト構築・チューブ作成
# ───────────────────────────────────────────
r       = fillet_r
off_mag = 0.05
delta_z = 0.1  # 交差時のZオフセット

mat_debug_inter = bpy.data.materials.new(name="DebugRed")
mat_debug_inter.diffuse_color = (1.0, 0.0, 0.0, 1.0)
mat_debug_edge  = bpy.data.materials.new(name="DebugBlue")
mat_debug_edge.diffuse_color  = (0.0, 0.0, 1.0, 1.0)

mat_v = bpy.data.materials.new(name="LabelRed")
mat_v.diffuse_color = (1.0, 0.0, 0.0, 1.0)
mat_e = bpy.data.materials.new(name="LabelBlue")
mat_e.diffuse_color = (0.0, 0.0, 1.0, 1.0)
edge_label_offset = 0.08

empties = {{}}

for route_idx, route in enumerate(vertex_edge_routes):
    # --- route構造チェック ---
    if len(route) < 3 or len(route) % 2 == 0:
        print(f"WARNING: route[{{route_idx}}] の長さ {{len(route)}} は奇数でなければなりません（頂点→辺→頂点→...）")
        #continue
    # 頂点→辺→頂点→辺... の構造になっているか簡易チェック
    structure_ok = True
    for i, key in enumerate(route):
        if i % 2 == 0:
            # 偶数番目: 頂点(数値またはstr)
            pass
        else:
            # 奇数番目: 辺(数値またはstr)
            if str(key) not in eid2info:
                print(f"WARNING: route[{{route_idx}}] の辺ID {{key}} が eid2info に存在しません")
                structure_ok = False
    #if not structure_ok:
    #    continue

    pts      = []
    pt_types = []

    L = len(route)
    for i, key in enumerate(route):
        if i % 2 == 0:
            # 頂点
            v = str(key)
            if v not in coords:
                print(f"WARNING: route[{{route_idx}}] の頂点ID {{v}} が coords に存在しません")
                continue
            x, y = coords[v]
            if i + 1 < len(route):
                eid_next = str(route[i+1])
            else:
                # cyclicならroute[1]だが、今回は最後は始点に戻さない（cyclic設定次第）
                eid_next = str(route[1]) if L > 1 else None
            z_flag_key = f"{{eid_next}}-{{v}}"
            sign = z_flags.get(z_flag_key, 1)
            if v == str(flat_vertex):
                z = 0.0
            else:
                z = sign * delta_z
            P = mathutils.Vector((x, y, z))
            pts.append(P)
            pt_types.append('P')
        else:
            # 辺制御点
            eid = str(key)
            if eid not in eid2info:
                print(f"WARNING: route[{{route_idx}}] の辺ID {{eid}} が eid2info に存在しません")
                continue
            u, v_ = eid2info[eid]
            mu, mv = coords[str(u)], coords[str(v_)]
            Mx, My = (mu[0] + mv[0]) * 0.5, (mu[1] + mv[1]) * 0.5
            M = mathutils.Vector((Mx, My, 0.0))
            sign = - preferred_signs.get(eid, 0)
            dx, dy = mv[0] - mu[0], mv[1] - mu[1]
            perp = mathutils.Vector((-dy, dx, 0.0)).normalized()
            M += perp * off_mag * sign
            pts.append(M)
            pt_types.append('M')


    is_cyclic = True


    # ───────────────────────────────────────────
    # 全ての制御点に小さな球を配置 & 色分け
    # ───────────────────────────────────────────
    mat_debug_inter = bpy.data.materials.new(name="DebugRed")
    mat_debug_inter.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    mat_debug_edge  = bpy.data.materials.new(name="DebugBlue")
    mat_debug_edge.diffuse_color  = (0.0, 0.0, 1.0, 1.0)

    # --- Bezier Curve 作成 ---
    curve_data = bpy.data.curves.new(f'GraphTube_{{route_idx}}', 'CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('BEZIER')
    # --- ここで n_ptsチェック＆安全なbezier_points追加 ---
    n_pts = len(pts)
    print("len(pts):", n_pts)
    if n_pts == 0:
        print(f"WARNING: pts is empty for route_idx={{route_idx}}")
        #continue   # このルートをスキップ
    if n_pts > 1:
        spline.bezier_points.add(n_pts - 1)  # もともと1点あるので-1
    print("len(spline.bezier_points):", len(spline.bezier_points))
    #assert len(spline.bezier_points) == n_pts

    #spline.bezier_points.add(len(pts) - 1)
    for idx, p in enumerate(pts):
        bp = spline.bezier_points[idx]
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    spline.use_cyclic_u = is_cyclic
    curve_data.resolution_u     = 16
    curve_data.bevel_depth      = 0.01
    curve_data.bevel_resolution = 32

    tube = bpy.data.objects.new(f'GraphTube_{{route_idx}}', curve_data)
    bpy.context.collection.objects.link(tube)

    # 1. チューブをアクティブ＆編集モードへ
    bpy.context.view_layer.objects.active = tube
    tube.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    
    vertex_counter = 0
    debug_r = sphere_r * 0.5
    for idx, (p, kind) in enumerate(zip(pts, pt_types)):
        # 2. 一度オブジェクトモードに戻って球・エンプティ追加
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = None

        if kind == 'P':
            v_id = str(route[vertex_counter * 2])   # 偶数番目が必ず頂点
            orient = 'U' if p.z >  0 else 'D' if p.z <  0 else 'F'
            empty_name = f"V{{v_id}}_{{orient}}_{{route_idx}}"
        else:
            eid = str(route[idx])
            empty_name = f"E{{eid}}_{{route_idx}}"
  
   

        empty = bpy.data.objects.new(empty_name, None)
        #empty = bpy.data.objects.new(f'Empty_{{route_idx}}_{{idx}}', None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.location = (p.x, p.y, p.z)
        bpy.context.collection.objects.link(empty)
        
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=debug_r,
            location=(p.x, p.y, p.z),
            segments=16,
            ring_count=8
        )

        ball = bpy.context.active_object
        mat  = mat_debug_inter if kind == 'P' else mat_debug_edge
        ball.data.materials.append(mat)

        if kind == 'P':
            ball.data.materials.append(mat_debug_inter)
        else:
            ball.data.materials.append(mat_debug_edge)

        empties[(route_idx, idx)] = empty     # route_idx, idx で引けるように

        ball.parent = empty
        ball.location = (0, 0, 0)

        if kind == 'P':
           empties[('V', v_id, orient, route_idx)] = empty
           empties[str(v_id)] = empty          # ←頂点ラベル用に従来キーも残す
           vertex_counter += 1
        else:
           empties[('E', eid,  route_idx)] = empty
           empties[(route_idx, idx)] = empty

        # 3. 再度tubeをアクティブに戻して編集モード
        bpy.context.view_layer.objects.active = tube
        tube.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')

        # 4. フックモディファイア追加
        hook = tube.modifiers.new(name=f"Hook_{{route_idx}}_{{idx}}", type='HOOK')
        hook.object = empty
        hook.strength = 1.0

        # 5. 制御点選択＆フック割当
        spline = tube.data.splines[0]
        for bp in spline.bezier_points:
            bp.select_control_point = False
        spline.bezier_points[idx].select_control_point = True
        bpy.ops.object.hook_assign(modifier=hook.name)
        #spline.bezier_points[idx].select_control_point = False

    # 最後はオブジェクトモードに戻す
    bpy.ops.object.mode_set(mode='OBJECT')
    tube.select_set(False)

    # --- ★ 輪っかごとに色を分ける ---
    if route_idx == 0:
        mat_loop = bpy.data.materials.new(name="LoopColor1")
        mat_loop.diffuse_color = (1.0, 1.0, 1.0, 1.0)  # 白、アルファ1
    elif route_idx == 1:
        mat_loop = bpy.data.materials.new(name="LoopColor2")
        mat_loop.diffuse_color = (1.0, 0.5, 0.5, 1.0)  # 薄い赤
    else:
        mat_loop = bpy.data.materials.new(name=f"LoopColor{{route_idx+1}}")
        mat_loop.diffuse_color = (0.8, 0.8, 1.0, 1.0)  # 薄い青など

    tube.data.materials.append(mat_loop)

    # --- メッシュ変換 & スムーズシェーディング ---
    #bpy.context.view_layer.objects.active = tube
    #tube.select_set(True)
    #bpy.ops.object.convert(target='MESH')
    #tube = bpy.context.active_object
    #mesh = tube.data
    #for poly in mesh.polygons:
    #    poly.use_smooth = True

    # --- 辺ラベル配置（各ルートごとに） ---
    for i, key in enumerate(route):
        if i % 2 == 1:
            eid = str(key)
            Mx, My, Mz = pts[i].x, pts[i].y, pts[i].z
            txt_curve = bpy.data.curves.new(name=f"E{{eid}}_{{route_idx}}", type='FONT')
            txt_curve.body = str(eid)
            txt_obj = bpy.data.objects.new(name=f"E{{eid}}_{{route_idx}}", object_data=txt_curve)
            txt_obj.location = (Mx, My, Mz + edge_label_offset)
            txt_obj.scale    = (0.1, 0.1, 0.1)
            txt_obj.data.materials.append(mat_e)
            bpy.context.collection.objects.link(txt_obj)

            par = empties.get((route_idx, i))
            if par:
                txt_obj.parent = par
                # Empty 原点に一致させる
                txt_obj.location = (0, 0, edge_label_offset)

# --- 頂点ラベルは全体共通で一回だけ ---

for v, coord in coords.items():
    txt_curve = bpy.data.curves.new(name=f"V{{v}}", type='FONT')
    txt_curve.body = str(v)
    txt_obj = bpy.data.objects.new(name=f"V{{v}}", object_data=txt_curve)
    txt_obj.location = (coord[0], coord[1], sphere_r + 0.1)
    txt_obj.scale    = (0.2, 0.2, 0.2)
    txt_obj.data.materials.append(mat_v)
    bpy.context.collection.objects.link(txt_obj)

    par = empties.get(str(v))   # kind=='P' のとき登録済み
    if par:
        
        txt_obj.parent = par
        txt_obj.location = (0, 0, sphere_r + 0.1)

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

#label = data.get("label", None)
if label is not None:
    # 上側のY座標を探す
    max_y = max(v[1] for v in coords.values())
    # 中央のX座標も使う（平均またはmax_yを持つ頂点のx座標）
    upper_verts = [k for k,v in coords.items() if v[1] == max_y]
    if upper_verts:
        x_pos = sum(coords[k][0] for k in upper_verts)/len(upper_verts)
    else:
        x_pos = 0.0
    # Y位置をさらにちょっと上へ
    y_pos = max_y + 0.3  # 見た目に応じて調整

    # 3Dテキスト追加
    txt_curve = bpy.data.curves.new(name="DiagramLabel", type='FONT')
    txt_curve.body = label
    txt_obj = bpy.data.objects.new(name="DiagramLabel", object_data=txt_curve)
    txt_obj.location = (x_pos, y_pos, 0.0)
    txt_obj.scale    = (0.2, 0.2, 0.2)
    # 緑のマテリアルを作って割当
    mat_label = bpy.data.materials.new(name="LabelGreen")
    mat_label.diffuse_color = (0.0, 1.0, 0.0, 1.0)
    txt_obj.data.materials.append(mat_label)
    bpy.context.collection.objects.link(txt_obj)
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
    # 輪っかの配列形式に変更
    vertex_edge_routes = data.get('vertex_edge_routes', [])
    eid2info_entries   = make_entries(data['eid2verts'])
    flat_vertex        = f'"{data["flat_vertex"]}"' if isinstance(data['flat_vertex'], str) else data['flat_vertex']
    z_flags_entries    = make_entries(data['z_flags'])
    pref_entries       = make_entries(data['preferred_signs'])
    label_entry = repr(data.get("label", None))

    script = TEMPLATE.format(
        coords_entries     = coords_entries,
        vertex_edge_routes = json.dumps(vertex_edge_routes, ensure_ascii=False, indent=2),
        eid2info_entries   = eid2info_entries,
        flat_vertex        = flat_vertex,
        z_flags_entries    = z_flags_entries,
        pref_entries       = pref_entries,
        label              = label_entry
    )

    with open('blender_script.py','w',encoding='utf-8') as f:
        f.write(script)
    print("生成: blender_script.py")

if __name__ == '__main__':
    main()
