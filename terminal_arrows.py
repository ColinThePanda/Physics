from textual.app import App, ComposeResult
from textual.widgets import Static
from textual_canvas import Canvas
from textual.reactive import reactive
from textual.color import Color
from linear_algebra.vector import vec2
from linear_algebra.matrix import rotation_matrix_2d, Matrix, shear_matrix_2d
import math
import numpy as np


def vector_to_angle(vector: vec2) -> float:
    angle_rad = math.atan2(vector.y, vector.x)
    return math.degrees(angle_rad) % 360


def draw_arrow_line(
    canvas: Canvas, color: Color, start: tuple[float, float], end: tuple[float, float]
) -> None:
    canvas.draw_line(
        int(start[0]), int(start[1]), int(end[0]), int(end[1]), color, refresh=False
    )


class VectorDisplay(Canvas):
    mouse_x: reactive[int] = reactive(0)
    mouse_y: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(width=50, height=50, **kwargs)
        self.center_x: int = 40
        self.center_y: int = 20
        self.mouse_x = self.center_x
        self.mouse_y = self.center_y

    def resize_canvas(self, width: int, height: int) -> None:
        self.clear(width=width, height=height)
        self.center_x = width // 2
        self.center_y = height // 2
        self.mouse_x = self.center_x
        self.mouse_y = self.center_y
        self.redraw_vectors()

    def redraw_vectors(self) -> None:
        with self.batch_refresh():
            self.clear()
            self.set_pixel(self.center_x, self.center_y, Color.parse("white"))
            direction = vec2(self.mouse_x - self.center_x, self.mouse_y - self.center_y)
            if direction.magnitude == 0:
                return
            vector_length = min(self.width, self.height) // 6
            normalized = direction.normalize() * vector_length

            end_pos = (self.center_x + normalized.x, self.center_y + normalized.y)
            draw_arrow_line(
                self, Color.parse("red"), (self.center_x, self.center_y), end_pos
            )

            rot = rotation_matrix_2d(np.pi / 2)
            rotated_vector = rot * normalized
            rotated_end = (
                self.center_x + rotated_vector.x,
                self.center_y + rotated_vector.y,
            )
            draw_arrow_line(
                self, Color.parse("green"), (self.center_x, self.center_y), rotated_end
            )

            scale_matrix = Matrix([[0.5, 0], [0, 1.5]])
            scaled_vector = scale_matrix * normalized
            scaled_end = (
                self.center_x + scaled_vector.x,
                self.center_y + scaled_vector.y,
            )
            draw_arrow_line(
                self, Color.parse("blue"), (self.center_x, self.center_y), scaled_end
            )

            shear_matrix = shear_matrix_2d(0.5, 0.5)
            sheared_vector = shear_matrix * normalized
            shear_end = (
                self.center_x + sheared_vector.x,
                self.center_y + sheared_vector.y,
            )
            draw_arrow_line(
                self, Color.parse("yellow"), (self.center_x, self.center_y), shear_end
            )

            if hasattr(self.app, "info_panel"):
                self.app.info_panel.update_info(
                    normalized, rotated_vector, scaled_vector, sheared_vector
                )


class InfoPanel(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.update_info(vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 0))

    def update_info(
        self, normal: vec2, rotated: vec2, scaled: vec2, sheared: vec2
    ) -> None:
        if normal.magnitude == 0:
            return
        normal_angle = round(vector_to_angle(normal.normalize()))
        rotated_angle = round(vector_to_angle(rotated.normalize()))
        scaled_angle = round(vector_to_angle(scaled.normalize()))
        sheared_angle = round(vector_to_angle(sheared.normalize()))

        info_text = f"""[bold red]Original Vector:[/bold red] {normal_angle}°
[bold green]Rotated (90°):[/bold green] {rotated_angle}°
[bold blue]Scaled (0.5x, 1.5y):[/bold blue] {scaled_angle}°
[bold yellow]Sheared (0.5, 0.5):[/bold yellow] {sheared_angle}°

[dim]Move your mouse around the canvas to control the base vector direction.[/dim]
[dim]Red = Original, Green = Rotated, Blue = Scaled, Yellow = Sheared[/dim]

[dim]Length: {round(normal.magnitude, 1)} pixels[/dim]"""

        self.update(info_text)


class VectorMatrixApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    #canvas {
        width: 3fr;
        height: 100%;
        border: solid white;
    }
    #info {
        width: 1fr;
        height: 100%;
        padding: 1;
        border: solid cyan;
        background: $panel;
    }
    #title {
        dock: top;
        height: 3;
        text-align: center;
        background: blue;
        color: white;
        padding: 1;
    }
    """

    last_canvas_region = None

    def compose(self) -> ComposeResult:
        yield Static("[bold]Vector and Matrix Transformations Demo[/bold]", id="title")
        self.canvas = VectorDisplay(id="canvas")
        yield self.canvas
        self.info_panel = InfoPanel(id="info")
        yield self.info_panel

    def on_mount(self) -> None:
        self.call_later(self.update_canvas_size)
        self.set_interval(0.1, self.check_canvas_resize)

    def on_resize(self, event) -> None:
        self.call_later(self.update_canvas_size)

    def update_canvas_size(self) -> None:
        self.refresh(layout=True)
        canvas_region = self.canvas.region
        if canvas_region.width > 0 and canvas_region.height > 0:
            width = canvas_region.width
            height = canvas_region.height * 2
            self.canvas.resize_canvas(width, height)
            self.last_canvas_region = canvas_region

    def on_mouse_move(self, event) -> None:
        canvas_region = self.canvas.region
        relative_x = max(
            0, min(event.screen_offset.x - canvas_region.x, canvas_region.width - 1)
        )
        relative_y = (
            max(
                0,
                min(event.screen_offset.y - canvas_region.y, canvas_region.height - 1),
            )
            * 2
        )
        self.canvas.mouse_x = relative_x
        self.canvas.mouse_y = relative_y
        self.canvas.redraw_vectors()

    def check_canvas_resize(self) -> None:
        canvas_region = self.canvas.region
        if (
            self.last_canvas_region is None
            or canvas_region.width != self.last_canvas_region.width
            or canvas_region.height != self.last_canvas_region.height
        ):
            self.update_canvas_size()


if __name__ == "__main__":
    app = VectorMatrixApp()
    app.run()
