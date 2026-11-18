import itertools
import numpy as np
from image_lab.validation import Invalid
from beadbench.plan import (
    create_plan, run_plan, failed_run, edit_plan, validate_plan,
    plan_summary, render_plan, BLANK,
)
from beadbench.core import assign, DEFAULT_PALETTE


def make_pattern(h=4, w=4, k=3, seed=1, locks=None):
    rng = np.random.default_rng(seed)
    palette = DEFAULT_PALETTE[:k]
    from beadbench.core import palette_rgb
    rgb = palette_rgb(palette)
    targets = rng.random((h, w, 3)).astype(np.float32)
    cells = assign(targets, palette)
    lk = np.zeros((h, w), bool) if locks is None else np.asarray(locks, bool)
    return dict(id='a' * 32, revision=0, name='t', width=w, height=h, palette=palette,
                targets=targets.tolist(), cells=cells.tolist(),
                locks=lk.tolist(), sampling='area', notes='')


def brute_best(targets, palette, locks, cells, stocks):
    """穷举所有未锁格配色，验证MILP得到全局最优（小图）。"""
    from beadbench.core import palette_rgb
    rgb = palette_rgb(palette)
    k = len(palette)
    targets = np.asarray(targets, np.float64)
    locks = np.asarray(locks, bool)
    cells = np.asarray(cells).copy()
    free = list(map(tuple, np.argwhere(~locks)))
    lock_used = np.bincount(cells[locks], minlength=k)
    best = None
    for combo in itertools.product(range(k), repeat=len(free)):
        used = lock_used.copy()
        ok = True
        for (y, x), c in zip(free, combo):
            used[c] += 1
            if used[c] > stocks[c]:
                ok = False
                break
        if not ok:
            continue
        trial = cells.copy()
        for (y, x), c in zip(free, combo):
            trial[y, x] = c
        err = float(((targets - rgb[trial]) ** 2).sum())
        if best is None or err < best[0] - 1e-12:
            best = (err, trial)
    return best


def test_fill_optimal_small():
    pat = make_pattern(4, 4, 3, seed=2)
    # 锁定12格，仅4格参与优化，便于穷举比对全局最优
    lk = np.zeros((4, 4), bool); lk.flat[:12] = True
    pat['locks'] = lk.tolist()
    stocks = [8, 8, 8]
    pl = create_plan(pat, dict(stocks=stocks, mode='fill'))
    out, rec = run_plan(pl, stocks, 'fill')
    s = plan_summary(out)
    assert s['blanks'] == 0 and s['filled'] == 16
    for c in s['colors']:
        assert c['used'] <= c['stock']
        assert c['used'] + c['remain'] == c['stock']
    best = brute_best(pat['targets'], pat['palette'], pat['locks'], np.asarray(pat['cells']), np.array(stocks))
    assert abs(s['squared_rgb_error'] - best[0]) < 1e-6
    assert s['total'] == 16


def test_fill_respects_locks_and_stocks():
    pat = make_pattern(4, 4, 4, seed=3)
    # 手工把(0,0)锁成色2
    pat['cells'][0][0] = 2
    pat['locks'][0][0] = True
    pl = create_plan(pat, dict(stocks=[4, 4, 4, 4], mode='fill'))
    out, _ = run_plan(pl, [4, 4, 4, 4], 'fill')
    assert out['cells'][0][0] == 2 and out['locks'][0][0]
    # 色2库存只给1：锁格占1后，其他格不能再用色2
    out, _ = run_plan(pl, [6, 6, 1, 6], 'fill')
    s = plan_summary(out)
    assert s['colors'][2]['used'] == 1
    assert out['cells'][0][0] == 2
    assert s['blanks'] == 0


def test_fill_lock_overstock_fails_and_keeps_previous():
    pat = make_pattern(4, 4, 2, seed=4)
    pat['cells'][0][0] = 0
    pat['locks'][0][0] = True
    pl = create_plan(pat, dict(stocks=[8, 8], mode='fill'))
    good, _ = run_plan(pl, [8, 8], 'fill')
    before = [row[:] for row in good['cells']]
    try:
        run_plan(good, [0, 9], 'fill')
        assert False
    except Invalid as e:
        assert '锁格已超过库存' in str(e)
    # 失败不改格
    assert good['cells'] == before
    assert good['runs'][-1]['ok'] is True


def test_fill_total_shortage():
    pat = make_pattern(4, 4, 2, seed=5)
    pl = create_plan(pat, dict(stocks=[2, 2], mode='fill'))
    try:
        run_plan(pl, [2, 2], 'fill')
        assert False
    except Invalid as e:
        assert '物料总数不足' in str(e)
    assert pl['cells'] == pat['cells']  # 初始未被破坏


def test_keep_keeps_original_colors_only():
    pat = make_pattern(4, 4, 3, seed=6)
    pl = create_plan(pat, dict(stocks=[1, 1, 100], mode='keep'))
    out, rec = run_plan(pl, [1, 1, 100], 'keep')
    cells = np.asarray(out['cells']); orig = np.asarray(out['original_cells'])
    filled = cells != BLANK
    # 所有填色格必须等于原配色
    assert np.all(cells[filled] == orig[filled])
    s = plan_summary(out)
    assert s['colors'][0]['used'] <= 1 and s['colors'][1]['used'] <= 1
    assert s['blanks'] >= 1
    # 每色保留数 = min(原需求, 库存)
    demand = np.bincount(orig.ravel(), minlength=3)
    for i in range(3):
        assert s['colors'][i]['used'] == min(int(demand[i]), [1, 1, 100][i])


def test_keep_with_locks_prioritizes_lock_then_fills():
    pat = make_pattern(4, 4, 2, seed=7)
    # 把(0,0)原是色c0的格锁为色1
    pat['cells'][0][0] = 1
    pat['locks'][0][0] = True
    pl = create_plan(pat, dict(stocks=[1, 1], mode='keep'))
    out, _ = run_plan(pl, [1, 1], 'keep')
    cells = np.asarray(out['cells'])
    assert cells[0, 0] == 1  # 锁格保留
    s = plan_summary(out)
    assert s['colors'][1]['used'] == 1
    # 未锁填色仍是原配色
    orig = np.asarray(out['original_cells']); locks = np.asarray(out['locks'])
    free_filled = (cells != BLANK) & ~locks
    assert np.all(cells[free_filled] == orig[free_filled])


def test_keep_lock_overstock():
    pat = make_pattern(4, 4, 2, seed=8)
    pat['cells'][0][0] = 0
    pat['locks'][0][0] = True
    pl = create_plan(pat, dict(stocks=[5, 5], mode='keep'))
    good, _ = run_plan(pl, [5, 5], 'keep')
    try:
        run_plan(good, [0, 9], 'keep')
        assert False
    except Invalid as e:
        assert '锁格已超过库存' in str(e)


def test_switch_modes_recompute_uses_original_not_last_result():
    pat = make_pattern(4, 4, 3, seed=9)
    pl = create_plan(pat, dict(stocks=[2, 2, 100], mode='keep'))
    kept, _ = run_plan(pl, [2, 2, 100], 'keep')
    # 切到fill，空格应能被换成其他色排满，依据targets而非空格
    filled, _ = run_plan(kept, [4, 4, 8], 'fill')
    s = plan_summary(filled)
    assert s['blanks'] == 0
    # 再切回keep，仍按original_cells（而非fill的替代色）
    again, _ = run_plan(filled, [2, 2, 100], 'keep')
    cells = np.asarray(again['cells']); orig = np.asarray(again['original_cells'])
    f = cells != BLANK
    assert np.all(cells[f] == orig[f])


def test_paint_then_lock_and_blank_lock_rejected():
    pat = make_pattern(4, 4, 2, seed=10)
    pl = create_plan(pat, dict(stocks=[1, 100], mode='keep'))
    out, _ = run_plan(pl, [1, 100], 'keep')
    assert (np.asarray(out['cells']) == BLANK).any()
    # 空格直接锁定应拒绝
    blank_xy = tuple(np.argwhere(np.asarray(out['cells']) == BLANK)[0])
    try:
        edit_plan(out, dict(mode='lock', changes=[[blank_xy[1], blank_xy[0]]]))
        assert False
    except Invalid as e:
        assert '先上色才能锁定' in str(e)
    # 先上色再锁定
    edit_plan(out, dict(mode='paint', color=1, changes=[[blank_xy[1], blank_xy[0]]]))
    edit_plan(out, dict(mode='lock', changes=[[blank_xy[1], blank_xy[0]]]))
    assert out['locks'][blank_xy[0]][blank_xy[1]]
    # 新锁格以指定颜色占料，重算fill时保留
    run_plan(out, [1, 100], 'fill')
    assert out['cells'][blank_xy[0]][blank_xy[1]] == 1
    assert plan_summary(out)['blanks'] == 0


def test_paint_and_lock_overstock_rejected():
    pat = make_pattern(4, 4, 2, seed=11)
    pl = create_plan(pat, dict(stocks=[100, 100], mode='fill'))
    out, _ = run_plan(pl, [100, 100], 'fill')
    # 找一个当前为色0的格，漆成色1才会增加色1用量
    yx = tuple(np.argwhere(np.asarray(out['cells']) == 0)[0])
    x0, y0 = int(yx[1]), int(yx[0])
    used1 = int(np.sum(np.asarray(out['cells']) == 1))
    # 色1库存压到恰好当前用量：再向色1漆一格即超库存，上色阶段拒绝且整体不生效
    out['stocks'] = [100, used1]
    before = [row[:] for row in out['cells']]
    try:
        edit_plan(out, dict(mode='paint', color=1, changes=[[x0, y0]]))
        assert False
    except Invalid as e:
        assert '手工上色后超过库存' in str(e)
    assert out['cells'] == before
    # 提高1颗库存后，漆色1并锁定允许（正好用满）
    out['stocks'] = [100, used1 + 1]
    edit_plan(out, dict(mode='paint', color=1, changes=[[x0, y0]]))
    edit_plan(out, dict(mode='lock', changes=[[x0, y0]]))
    assert out['locks'][y0][x0] and out['cells'][y0][x0] == 1
    # 再把另一色0格漆成色1又超库存
    yx2 = tuple(np.argwhere(np.asarray(out['cells']) == 0)[0])
    try:
        edit_plan(out, dict(mode='paint', color=1, changes=[[int(yx2[1]), int(yx2[0])]]))
        assert False
    except Invalid as e:
        assert '手工上色后超过库存' in str(e)
    assert not out['locks'][int(yx2[0])][int(yx2[1])]


def test_lock_overstock_edit_rejected():
    # 库存足够上色，但锁格统计超限时锁定仍被拒绝
    pat = make_pattern(4, 4, 2, seed=11)
    pl = create_plan(pat, dict(stocks=[100, 100], mode='fill'))
    out, _ = run_plan(pl, [100, 100], 'fill')
    edit_plan(out, dict(mode='paint', color=1, changes=[[0, 0]]))
    # 随后把色1库存清零：该格再锁定即超锁格库存
    out['stocks'] = [100, 0]
    try:
        edit_plan(out, dict(mode='lock', changes=[[0, 0]]))
        assert False
    except Invalid as e:
        assert '锁定后超过库存' in str(e)
    # 编辑整体未生效
    assert not out['locks'][0][0]


def test_unlock_then_recompute():
    pat = make_pattern(4, 4, 2, seed=12)
    pat['locks'][0][0] = True
    pl = create_plan(pat, dict(stocks=[9, 9], mode='fill'))
    out, _ = run_plan(pl, [9, 9], 'fill')
    edit_plan(out, dict(mode='unlock', changes=[[0, 0]]))
    assert not out['locks'][0][0]


def test_summary_blanks_not_counted_as_material():
    pat = make_pattern(4, 4, 2, seed=13)
    pl = create_plan(pat, dict(stocks=[1, 1], mode='keep'))
    out, _ = run_plan(pl, [1, 1], 'keep')
    s = plan_summary(out)
    assert s['filled'] + s['blanks'] == 16
    assert sum(c['used'] for c in s['colors']) == s['filled']
    # 误差只统计有豆格
    rgb = np.asarray([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in pat['palette']])
    cells = np.asarray(out['cells']); targets = np.asarray(out['targets'])
    f = cells != BLANK
    expect = float(((targets[f] - rgb[cells[f]]) ** 2).sum())
    assert abs(s['squared_rgb_error'] - expect) < 1e-6


def test_failed_run_recorded():
    pat = make_pattern(4, 4, 2, seed=14)
    pl = create_plan(pat, dict(stocks=[2, 2], mode='fill'))
    try:
        run_plan(pl, [2, 2], 'fill')
    except Invalid as e:
        failed_run(pl, [2, 2], 'fill', e)
    validate_plan(pl)
    assert pl['runs'][-1]['ok'] is False and '物料总数不足' in pl['runs'][-1]['reason']


def test_render_has_blank_marks_and_png():
    pat = make_pattern(4, 4, 2, seed=15)
    pl = create_plan(pat, dict(stocks=[1, 1], mode='keep'))
    run_plan(pl, [1, 1], 'keep')
    raw = render_plan(pl)
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'


def test_stock_validation():
    pat = make_pattern(4, 4, 2, seed=16)
    try:
        create_plan(pat, dict(stocks=[1], mode='fill'))
        assert False
    except Invalid:
        pass
    try:
        create_plan(pat, dict(stocks=[-1, 2], mode='fill'))
        assert False
    except Invalid:
        pass
    try:
        create_plan(pat, dict(stocks=[2, 2], mode='wat'))
        assert False
    except Invalid:
        pass


def test_large_fill_performance_and_feasibility():
    import time
    pat = make_pattern(32, 32, 8, seed=17)
    pl = create_plan(pat, dict(stocks=[128] * 8, mode='fill'))
    t = time.time()
    out, _ = run_plan(pl, [128] * 8, 'fill')
    assert time.time() - t < 5
    s = plan_summary(out)
    assert s['blanks'] == 0 and s['filled'] == 1024
    assert all(c['used'] <= 128 for c in s['colors'])
