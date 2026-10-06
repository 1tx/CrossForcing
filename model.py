# -*- coding: utf-8 -*-
"""
模型定义。

主方法 DeltaOperator（显式状态神经算子）：
    state_t = clamp( state_{t-1} + MLP([state_{t-1}, F_t, static]) )
  时间约定：forcing[t] 表示从 state[t-1] 推进到 state[t] 这一时段（第 t 天）的大气强迫，
  因此 state_t = model(state_{t-1}, forcing_t, static)；训练 / one-step / free-run 统一此约定。
  量纲统一在模型内部完成（LayerNorm / BatchNorm），输入保持原始物理单位，
  状态以原始 m3/m3 递推（显式物理状态，可保存重启、可内嵌水量守恒）。

baseline LSTMState（隐状态 LSTM，lookback 窗口 -> 下一时刻状态）。
"""
import torch
import torch.nn as nn

import config


def _act(name):
    return nn.ReLU() if name == "relu" else nn.Tanh()


def _norm(dim, kind):
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class DeltaOperator(nn.Module):
    """显式状态算子：预测增量 ΔSM，再与状态相加并物理裁剪。"""

    def __init__(self, mc=None):
        super().__init__()
        mc = mc or config.cfg.model
        self.state_clip = mc.state_clip
        in_dim = mc.n_state + mc.n_forcing + mc.n_static

        layers = []
        if mc.input_norm:
            layers.append(_norm(in_dim, mc.norm))
        d = in_dim
        for _ in range(mc.n_layers):
            layers.append(nn.Linear(d, mc.hidden))
            layers.append(_norm(mc.hidden, mc.norm))
            layers.append(_act(mc.activation))
            d = mc.hidden
        layers.append(nn.Linear(d, mc.n_state))
        self.net = nn.Sequential(*layers)

    def forward(self, state, forcing, static):
        x = torch.cat([state, forcing, static], dim=-1)
        delta = self.net(x)
        nxt = state + delta
        if self.state_clip is not None:
            nxt = torch.clamp(nxt, self.state_clip[0], self.state_clip[1])
        return nxt, delta


class LSTMState(nn.Module):
    """baseline：隐状态 LSTM，输入 lookback 窗口的 [state, forcing, static]，预测下一时刻 ΔSM。"""

    def __init__(self, mc=None, lookback=7):
        super().__init__()
        mc = mc or config.cfg.model
        self.lookback = lookback
        self.state_clip = mc.state_clip
        in_dim = mc.n_state + mc.n_forcing + mc.n_static
        # LSTM 输入是 (B, L, D)，只支持 LayerNorm/None（BatchNorm1d 需 (B,C,L) 转置，暂不支持）
        self.in_norm = _norm(in_dim, mc.norm if mc.norm != "batch" else "layer")
        self.lstm = nn.LSTM(in_dim, mc.hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(mc.hidden, mc.n_state)

    def forward(self, states, forcings, static):
        # states:(B,L,2)  forcings:(B,L,F)  static:(B,S)
        s = torch.cat([states, forcings, static.unsqueeze(1).expand(-1, states.size(1), -1)], dim=-1)
        s = self.in_norm(s)
        out, _ = self.lstm(s)
        delta = self.head(out[:, -1])
        nxt = states[:, -1] + delta
        if self.state_clip is not None:
            nxt = torch.clamp(nxt, self.state_clip[0], self.state_clip[1])
        return nxt, delta


def build_model(mc=None):
    """按 mc.version 构造模型："lstm" -> LSTMState，否则 DeltaOperator。"""
    mc = mc or config.cfg.model
    return LSTMState(mc, lookback=mc.lookback) if mc.version == "lstm" else DeltaOperator(mc)


def free_run(model, mc, target, forcing, static, t_start, n_steps):
    """自回归 free-run：从 t_start 起仅用 forcing 驱动 n_steps，返回 (N, n_steps, 2)。

    参数：
      target/forcing —— (N, T, *) 张量（须在 model 所在 device 上）；static —— (N, S)。
      t_start —— 起始日（绝对时间索引）；n_steps —— 步数。

    语义：
      delta：初始状态取 t_start-1 真值，逐点递推 state <- model(state, F_t, static)。
      lstm ：初始窗口取 [t_start-L, t_start) 真值，滑窗递推（每步丢最老、append 预测）。

    调用方需保证 t_start-L >= 0 且 t_start+n_steps <= T。
    """
    N = target.shape[0]
    if mc.version == "lstm":
        L = mc.lookback
        state_hist = target[:, t_start - L:t_start]              # (N, L, 2) 真值初始化
        out = torch.empty((N, n_steps, 2), device=target.device, dtype=target.dtype)
        arange_lb = torch.arange(L, device=target.device) - L    # [-L, ..., -1]
        for i in range(n_steps):
            t = t_start + i
            idx = t + arange_lb                                  # [t-L, ..., t-1]
            nxt, _ = model(state_hist, forcing[:, idx, :], static)
            out[:, i] = nxt
            state_hist = torch.cat([state_hist[:, 1:], nxt[:, None]], dim=1)
        return out
    # delta
    state = target[:, t_start - 1]                               # (N, 2)
    out = torch.empty((N, n_steps, 2), device=target.device, dtype=target.dtype)
    for i in range(n_steps):
        t = t_start + i
        state, _ = model(state, forcing[:, t], static)
        out[:, i] = state
    return out
