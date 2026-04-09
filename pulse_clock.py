#!/usr/bin/env python3
"""终端脉冲时钟。

持续刷新当前时间，并通过变化的条形长度模拟“脉冲”效果。
按 Ctrl+C 退出。
"""

from __future__ import annotations

import math
import time
from datetime import datetime

BAR_WIDTH = 24
PULSE_HZ = 0.75
FRAME_INTERVAL = 0.1


def render_pulse_bar(width: int = 24, phase: float = 0.0) -> str:
    """根据相位生成脉冲条。"""
    value = (math.sin(phase) + 1) / 2  # 0~1
    filled = max(1, int(round(value * width)))
    return "█" * filled + "·" * (width - filled)


def run_pulse_clock() -> None:
    """循环显示带脉冲效果的当前时间。"""
    print("脉冲时钟启动（Ctrl+C 退出）")
    print()

    start = time.monotonic()
    while True:
        frame_start = time.monotonic()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = frame_start - start
        phase = elapsed * 2 * math.pi * PULSE_HZ
        pulse = render_pulse_bar(width=BAR_WIDTH, phase=phase)
        line = f"\r🕒 {now}  |  {pulse}"
        print(line, end="", flush=True)
        remaining = FRAME_INTERVAL - (time.monotonic() - frame_start)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    try:
        run_pulse_clock()
    except KeyboardInterrupt:
        print("\n已退出脉冲时钟。")
