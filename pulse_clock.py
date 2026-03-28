#!/usr/bin/env python3
"""终端脉冲时钟。

每秒刷新一次当前时间，并通过变化的条形长度模拟“脉冲”效果。
按 Ctrl+C 退出。
"""

from __future__ import annotations

import math
import time
from datetime import datetime


def render_pulse_bar(width: int = 24, phase: float = 0.0) -> str:
    """根据相位生成脉冲条。"""
    value = (math.sin(phase) + 1) / 2  # 0~1
    filled = max(1, int(round(value * width)))
    return "█" * filled + "·" * (width - filled)


def run_pulse_clock() -> None:
    """循环显示带脉冲效果的当前时间。"""
    print("脉冲时钟启动（Ctrl+C 退出）")
    print()

    start = time.time()
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - start
        pulse = render_pulse_bar(phase=elapsed * math.pi)
        line = f"\r🕒 {now}  |  {pulse}"
        print(line, end="", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    try:
        run_pulse_clock()
    except KeyboardInterrupt:
        print("\n已退出脉冲时钟。")
