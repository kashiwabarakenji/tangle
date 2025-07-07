import tkinter as tk
import json

class DraggablePoint:
    def __init__(self, canvas, x, y, width, height, index, radius=15):
        self.canvas = canvas
        self.radius = radius
        self.width = width
        self.height = height
        self.index = index

        # 初期位置
        px, py = self.logic_to_pixel(x, y)

        # 丸と番号
        self.oval = canvas.create_oval(px-radius, py-radius, px+radius, py+radius, fill="lightblue")
        self.number = canvas.create_text(px, py, text=str(index), font=("Arial", 12, "bold"))

        # 座標（論理座標系）
        self.x_logic = x
        self.y_logic = y

        # ドラッグフラグ
        self.dragging = False

        # イベントバインド
        for tag in (self.oval, self.number):
            canvas.tag_bind(tag, "<ButtonPress-1>", self.on_press)
            canvas.tag_bind(tag, "<ButtonRelease-1>", self.on_release)
            canvas.tag_bind(tag, "<B1-Motion>", self.on_motion)

    def logic_to_pixel(self, x, y):
        px = (x + 1) / 2 * self.width
        py = (1 - y) / 2 * self.height
        return px, py

    def pixel_to_logic(self, px, py):
        x = px / self.width * 2 - 1
        y = 1 - py / self.height * 2
        return x, y

    def snap(self, val, step=0.1):
        return round(val / step) * step

    def on_press(self, event):
        self.dragging = True
        self.update_position(event.x, event.y)

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if self.dragging:
            self.update_position(event.x, event.y)

    def update_position(self, px, py):
        x_logic, y_logic = self.pixel_to_logic(px, py)
        x_logic = self.snap(x_logic, 0.1)
        y_logic = self.snap(y_logic, 0.1)

        px_snapped, py_snapped = self.logic_to_pixel(x_logic, y_logic)

        self.canvas.coords(self.oval,
                           px_snapped - self.radius,
                           py_snapped - self.radius,
                           px_snapped + self.radius,
                           py_snapped + self.radius)
        self.canvas.coords(self.number, px_snapped, py_snapped)

        self.x_logic = x_logic
        self.y_logic = y_logic

    def get_coords(self):
        return [self.x_logic, self.y_logic]

def main(num_points):
    root = tk.Tk()
    root.title("Draggable Numbered Points with File Output")
    width = 600
    height = 600
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack()

    points = []
    for i in range(num_points):
        x = -0.9 + i * 0.2
        y = 0.0
        point = DraggablePoint(canvas, x, y, width, height, index=i)
        points.append(point)

    def output_coords():
        data = {"coords": {}}
        for p in points:
            coords = p.get_coords()
            data["coords"][str(p.index)] = coords

        with open("location.json", "w") as f:
            json.dump(data, f, indent=2)

        print("座標データを location.json に出力しました。")

    btn = tk.Button(root, text="座標をファイル出力", command=output_coords)
    btn.pack()

    root.mainloop()

if __name__ == "__main__":
    main(7)  # 点の数を指定
