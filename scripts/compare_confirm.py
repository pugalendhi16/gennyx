#!/usr/bin/env python3
"""Compare UT Bot confirm=0 vs confirm=1 backtest results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import (
    BASE_CONFIG, SESSION_CONFIGS, fetch_data, add_indicators,
    align_htf, run_backtest, calc_metrics
)


def fmt_val(key, val):
    if key in ['total_trades', 'winning_trades', 'max_consec_wins', 'max_consec_losses']:
        return f'{int(val)}'
    elif key in ['win_rate', 'total_return', 'max_dd_pct']:
        return f'{val:.1f}%'
    elif key in ['total_pnl', 'avg_trade', 'avg_winner', 'avg_loser', 'largest_win', 'largest_loss', 'max_dd']:
        return f'${val:+,.2f}'
    elif key in ['profit_factor', 'sharpe']:
        return f'{val:.2f}'
    elif key == 'avg_hold_hrs':
        return f'{val:.1f}h'
    return str(val)


def fmt_diff(key, diff):
    if key in ['total_trades', 'winning_trades']:
        return f'{int(diff):+d}'
    elif key in ['win_rate', 'total_return', 'max_dd_pct']:
        return f'{diff:+.1f}%'
    elif key in ['total_pnl', 'avg_trade', 'avg_winner', 'avg_loser', 'largest_win', 'largest_loss', 'max_dd']:
        return f'${diff:+,.2f}'
    elif key in ['profit_factor', 'sharpe']:
        return f'{diff:+.2f}'
    elif key == 'avg_hold_hrs':
        return f'{diff:+.1f}h'
    return str(diff)


def main():
    cfg = BASE_CONFIG.copy()
    cap = cfg['initial_capital']

    print('=' * 90)
    print('COMPARISON: confirm=0 vs confirm=1 (TOS-style)'.center(90))
    print('=' * 90)
    print(f"  Symbol: {cfg['symbol']} | Capital: ${cap:,.0f} | Sensitivity: {cfg['ut_sensitivity']}")
    print()

    # Fetch data once
    df_5m_raw, df_1h_raw = fetch_data(cfg['symbol'], cfg['days'])
    print(f'  5m bars: {len(df_5m_raw):,}  |  1h bars: {len(df_1h_raw):,}')
    print()

    results = {}
    for confirm in [0, 1]:
        ovn_cfg = {**cfg, **SESSION_CONFIGS['overnight'], 'ut_confirmation_bars': confirm}

        print(f'  Running confirm={confirm} ...', end=' ', flush=True)
        df_5m = add_indicators(df_5m_raw, ovn_cfg)
        df_1h = add_indicators(df_1h_raw, ovn_cfg)
        htf = align_htf(df_5m, df_1h)
        trades, curve = run_backtest(df_5m, htf, ovn_cfg, 'overnight')
        m = calc_metrics(trades, curve, cap)
        results[confirm] = m
        print(f'done -- {len(trades)} trades')

    # Comparison table
    m0, m1 = results[0], results[1]

    rows = [
        ('Total Trades', 'total_trades'),
        ('Winning Trades', 'winning_trades'),
        ('Win Rate', 'win_rate'),
        ('Total P&L', 'total_pnl'),
        ('Total Return', 'total_return'),
        ('Profit Factor', 'profit_factor'),
        ('Avg Trade P&L', 'avg_trade'),
        ('Avg Winner', 'avg_winner'),
        ('Avg Loser', 'avg_loser'),
        ('Largest Win', 'largest_win'),
        ('Largest Loss', 'largest_loss'),
        ('Max Drawdown', 'max_dd'),
        ('Max Drawdown %', 'max_dd_pct'),
        ('Sharpe Ratio', 'sharpe'),
        ('Avg Holding Time', 'avg_hold_hrs'),
    ]

    print()
    print('=' * 90)
    print(f"{'Metric':<24} {'confirm=0 (original)':>22} {'confirm=1 (TOS)':>22} {'Difference':>20}")
    print('-' * 90)

    for name, key in rows:
        v0, v1 = m0[key], m1[key]
        s0 = fmt_val(key, v0)
        s1 = fmt_val(key, v1)
        diff = v1 - v0
        sdiff = fmt_diff(key, diff)
        print(f'{name:<24} {s0:>22} {s1:>22} {sdiff:>20}')

    print('=' * 90)
    print()

    # Recommendation
    pnl_diff = m1['total_pnl'] - m0['total_pnl']
    if pnl_diff > 0:
        print(f'RESULT: confirm=1 (TOS-style) is BETTER by ${pnl_diff:,.2f}')
    else:
        print(f'RESULT: confirm=0 (original) is BETTER by ${-pnl_diff:,.2f}')

    # Additional insights
    print()
    print('KEY INSIGHTS:')
    print(f'  - Trades: {m1["total_trades"] - m0["total_trades"]:+d} trades with TOS-style')
    print(f'  - Win Rate: {m1["win_rate"] - m0["win_rate"]:+.1f}% with TOS-style')
    print(f'  - Profit Factor: {m1["profit_factor"] - m0["profit_factor"]:+.2f} with TOS-style')
    print(f'  - Max Drawdown: {m1["max_dd_pct"] - m0["max_dd_pct"]:+.1f}% with TOS-style')


if __name__ == "__main__":
    main()
