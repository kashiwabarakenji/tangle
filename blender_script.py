import bpy, mathutils, math

# ───────────────────────────────────────────
# 0. シーンをクリア
# ───────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ───────────────────────────────────────────
# 1. 入力データ
# ───────────────────────────────────────────
coords = {
    "0": [-1.0, -0.3947368421052631],
    "1": [0.8421052631578947, -0.3947368421052631],
    "2": [-0.07894736842105267, 0.5263157894736842],
    "3": [-0.631578947368421, -0.21052631578947367],
    "4": [0.4736842105263157, -0.0263157894736842],
    "5": [0.1052631578947368, 0.34210526315789475],
    "6": [0.28947368421052627, 0.15789473684210525],
}

vertex_edge_routes = [
  [
    0,
    2,
    2,
    7,
    3,
    9,
    4,
    11,
    6,
    13,
    5,
    6,
    2,
    3,
    0,
    1,
    1,
    4,
    3,
    8,
    5,
    12,
    6,
    10,
    4,
    5,
    1,
    0
  ]
]

eid2info = {
    "0": [0, 1],
    "1": [0, 1],
    "2": [0, 2],
    "3": [0, 2],
    "4": [1, 3],
    "5": [1, 4],
    "6": [2, 5],
    "7": [2, 3],
    "8": [3, 5],
    "9": [3, 4],
    "10": [4, 6],
    "11": [4, 6],
    "12": [5, 6],
    "13": [5, 6],
}

flat_vertex = 3

z_flags = {
    "0-0": 1,
    "1-0": -1,
    "2-0": 1,
    "3-0": -1,
    "4-1": 1,
    "5-1": -1,
    "1-1": 1,
    "0-1": -1,
    "6-2": 1,
    "7-2": -1,
    "3-2": 1,
    "2-2": -1,
    "5-4": 1,
    "9-4": -1,
    "10-4": 1,
    "11-4": -1,
    "12-5": 1,
    "13-5": -1,
    "8-5": 1,
    "6-5": -1,
    "13-6": 1,
    "12-6": -1,
    "11-6": 1,
    "10-6": -1,
}

preferred_signs = {
    "0": -1,
    "1": 1,
    "2": -1,
    "3": 1,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": -1,
    "11": 1,
    "12": -1,
    "13": 1,
}

label = '6K7'

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

for route_idx, route in enumerate(vertex_edge_routes):
    # --- 制御点リストの生成 ---
    pts      = []
    pt_types = []

    L = len(route)
    # --- 制御点列生成 ---
    for i, key in enumerate(route):
        if i % 2 == 0:
            # 頂点
            v = str(key)
            x, y = coords[v]
            # 次に進む辺の情報
            if i + 1 < len(route):
                eid_next = str(route[i+1])
            else:
                eid_next = str(route[1])
            z_flag_key = f"{eid_next}-{v}"
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

    # --- Bezier Curve 作成 & チューブ化 ---
    curve_data = bpy.data.curves.new(f'GraphTube_{route_idx}', 'CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(pts) - 1)
    for idx, p in enumerate(pts):
        bp = spline.bezier_points[idx]
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'

    spline.use_cyclic_u = is_cyclic

    curve_data.resolution_u     = 16
    curve_data.bevel_depth      = 0.01
    curve_data.bevel_resolution = 32

    tube = bpy.data.objects.new(f'GraphTube_{route_idx}', curve_data)
    bpy.context.collection.objects.link(tube)

    # --- ★ 輪っかごとに色を分ける ---
    if route_idx == 0:
        mat_loop = bpy.data.materials.new(name="LoopColor1")
        mat_loop.diffuse_color = (1.0, 1.0, 1.0, 1.0)  # 白、アルファ1
    elif route_idx == 1:
        mat_loop = bpy.data.materials.new(name="LoopColor2")
        mat_loop.diffuse_color = (1.0, 0.5, 0.5, 1.0)  # 薄い赤
    else:
        mat_loop = bpy.data.materials.new(name=f"LoopColor{route_idx+1}")
        mat_loop.diffuse_color = (0.8, 0.8, 1.0, 1.0)  # 薄い青など

    tube.data.materials.append(mat_loop)

    # --- メッシュ変換 & スムーズシェーディング ---
    bpy.context.view_layer.objects.active = tube
    tube.select_set(True)
    bpy.ops.object.convert(target='MESH')
    tube = bpy.context.active_object
    mesh = tube.data
    for poly in mesh.polygons:
        poly.use_smooth = True

    # --- 辺ラベル配置（各ルートごとに） ---
    for i, key in enumerate(route):
        if i % 2 == 1:
            eid = str(key)
            Mx, My, Mz = pts[i].x, pts[i].y, pts[i].z
            txt_curve = bpy.data.curves.new(name=f"E{eid}_{route_idx}", type='FONT')
            txt_curve.body = str(eid)
            txt_obj = bpy.data.objects.new(name=f"E{eid}_{route_idx}", object_data=txt_curve)
            txt_obj.location = (Mx, My, Mz + edge_label_offset)
            txt_obj.scale    = (0.1, 0.1, 0.1)
            txt_obj.data.materials.append(mat_e)
            bpy.context.collection.objects.link(txt_obj)

# --- 頂点ラベルは全体共通で一回だけ ---
for v, coord in coords.items():
    txt_curve = bpy.data.curves.new(name=f"V{v}", type='FONT')
    txt_curve.body = str(v)
    txt_obj = bpy.data.objects.new(name=f"V{v}", object_data=txt_curve)
    txt_obj.location = (coord[0], coord[1], sphere_r + 0.1)
    txt_obj.scale    = (0.2, 0.2, 0.2)
    txt_obj.data.materials.append(mat_v)
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
