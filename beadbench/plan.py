import io
import numpy as np
from PIL import Image, ImageDraw
from image_lab.validation import Invalid
from .core import palette, palette_rgb, integer, validate_pattern

BLANK = -1
MAX_STOCK = 100000


def stock_value(value, name):
    return integer(value, name, 0, MAX_STOCK)


def stocks_value(value, k):
    if value is None:
        return [0] * k
    if not isinstance(value, list) or len(value) != k:
        raise Invalid(f'库存需为{k}个非负整数')
    return [stock_value(v, f'色号{chr(65 + i)}库存') for i, v in enumerate(value)]


def mode_value(value):
    if value not in ('fill', 'keep'):
        raise Invalid('排料方式需为排满或保原配色')
    return value


def validate_plan(p):
    colors = palette(p['palette'])
    w = integer(p['width'], '列数', 4, 32)
    h = integer(p['height'], '行数', 4, 32)
    k = len(colors)
    cells = np.asarray(p['cells'])
    locks = np.asarray(p['locks'])
    targets = np.asarray(p['targets'], np.float32)
    original = np.asarray(p['original_cells'])
    if cells.shape != (h, w) or locks.shape != (h, w) or targets.shape != (h, w, 3) or original.shape != (h, w):
        raise Invalid('排料图纸数组尺寸不一致')
    if not np.issubdtype(cells.dtype, np.integer) or np.any(cells < BLANK) or np.any(cells >= k):
        raise Invalid('格子的颜色编号不在色板内')
    if locks.dtype != np.bool_:
        raise Invalid('锁格状态需为布尔值')
    if not np.issubdtype(original.dtype, np.integer) or np.any(original < 0) or np.any(original >= k):
        raise Invalid('原配色编号不在色板内')
    if not np.isfinite(targets).all() or np.any(targets < 0) or np.any(targets > 1):
        raise Invalid('采样目标色无效')
    if np.any(locks & (cells == BLANK)):
        raise Invalid('空格不能锁定')
    stocks_value(p.get('stocks'), k)
    mode_value(p.get('mode'))
    if not isinstance(p.get('runs'), list) or len(p['runs']) > 200:
        raise Invalid('排料记录无效')
    for r in p['runs']:
        if not isinstance(r, dict) or r.get('mode') not in ('fill', 'keep') or not isinstance(r.get('ok'), bool):
            raise Invalid('排料记录无效')
        if not isinstance(r.get('stocks'), list) or len(r['stocks']) != k:
            raise Invalid('排料记录无效')
    return p


def create_plan(pattern, body):
    validate_pattern(pattern)
    k = len(pattern['palette'])
    cells = np.asarray(pattern['cells'])
    locks = np.asarray(pattern['locks'])
    # 普通图纸没有空格；来源锁格沿用其当前色
    if np.any(cells < 0):
        raise Invalid('来源图纸含空格，不能用于排料')
    stocks = stocks_value(body.get('stocks'), k)
    mode = mode_value(body.get('mode', 'fill'))
    h, w = cells.shape
    return validate_plan(dict(
        name=str(body.get('name', pattern['name'] + ' · 库存排料'))[:100],
        width=w, height=h, palette=list(pattern['palette']),
        targets=np.asarray(pattern['targets'], np.float32).tolist(),
        original_cells=cells.tolist(),
        cells=cells.tolist(),
        locks=locks.tolist(),
        stocks=stocks, mode=mode, runs=[], notes='',
        source=dict(
            pattern_id=pattern['id'], pattern_name=str(pattern['name'])[:100],
            pattern_revision=pattern['revision'], palette=list(pattern['palette']),
        ),
    ))


def _check_locks(cells, locks, stocks, k):
    used = np.bincount(cells[locks], minlength=k)
    over = [i for i in range(k) if used[i] > stocks[i]]
    if over:
        labels = '、'.join(f'{chr(65 + i)}需{int(used[i])}颗/库存{stocks[i]}颗' for i in over)
        raise Invalid('锁格已超过库存：' + labels + '；本次排料未执行，保留上次有效结果')
    return used


def _fill(targets, colors, cells, locks, stocks):
    """排满，允许换色：每格都有豆，未锁格在库存容量内最小化对采样目标RGB的总平方误差。"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    k = len(colors)
    h, w = cells.shape
    lock_used = _check_locks(cells, locks, np.asarray(stocks), k)
    free = np.argwhere(~locks)
    m = len(free)
    if m == 0:
        return cells.copy(), lock_used
    rgb = palette_rgb(colors)
    t = targets[free[:, 0], free[:, 1]]
    cost = np.sum((t[:, None, :] - rgb[None, :, :]) ** 2, axis=2).astype(np.float64)
    cap = np.maximum(np.asarray(stocks, dtype=int) - lock_used, 0)
    if int(cap.sum()) < m:
        raise Invalid(
            f'排满需要{m + int(lock_used.sum())}颗（未锁{m}颗），各色剩余可用库存合计仅{int(cap.sum())}颗，物料总数不足；'
            '本次排料未执行，保留上次有效结果。可改用“保原配色，缺料留空”或补充库存')
    rows, cols, data, lb, ub = [], [], [], [], []
    for r in range(m):
        for c in range(k):
            rows.append(r); cols.append(r * k + c); data.append(1)
        lb.append(1); ub.append(1)
    base = m
    for c in range(k):
        for r in range(m):
            rows.append(base + c); cols.append(r * k + c); data.append(-1)
        lb.append(-int(cap[c])); ub.append(np.inf)
    a = coo_matrix((data, (rows, cols)), shape=(m + k, m * k))
    res = milp(c=cost.reshape(-1), constraints=LinearConstraint(a, np.asarray(lb), np.asarray(ub)),
               integrality=np.ones(m * k), bounds=Bounds(0, 1))
    if res.status != 0 or res.x is None:
        raise Invalid('排满优化未能求得可行方案（库存容量无法覆盖所有格子）；本次排料未执行，保留上次有效结果')
    x = np.rint(res.x).astype(int).reshape(m, k)
    if np.any(x.sum(axis=1) != 1):
        raise Invalid('排满优化结果异常；本次排料未执行，保留上次有效结果')
    out = cells.copy()
    chosen = x.argmax(axis=1)
    for (yy, xx), c in zip(free, chosen):
        out[yy, xx] = int(c)
    used = np.bincount(out.ravel(), minlength=k)
    if np.any(used > np.asarray(stocks)):
        raise Invalid('排满结果超出库存；本次排料未执行，保留上次有效结果')
    return out, used


def _keep(original, cells, locks, stocks):
    """保原配色，缺料留空：未锁格只能沿用排料开始前的原配色，同色尽量多保留，其余明确留空。"""
    k = len(stocks)
    stocks = np.asarray(stocks, dtype=int)
    _check_locks(cells, locks, stocks, k)
    out = np.full_like(cells, BLANK)
    out[locks] = cells[locks]
    used = np.bincount(out[locks], minlength=k)
    # 同色格保留谁可任选：按行优先顺序确定性保留
    for y, x in np.argwhere(~locks):
        c = int(original[y, x])
        if used[c] < stocks[c]:
            out[y, x] = c
            used[c] += 1
    return out, used


def run_plan(p, stocks=None, mode=None):
    """按给定库存与方式重算；失败抛 Invalid 且不改动 p。成功才写入并追加排料记录。"""
    validate_plan(p)
    k = len(p['palette'])
    stocks = stocks_value(stocks if stocks is not None else p.get('stocks'), k)
    mode = mode_value(mode if mode is not None else p.get('mode'))
    cells = np.asarray(p['cells'])
    locks = np.asarray(p['locks'], dtype=bool)
    targets = np.asarray(p['targets'], np.float32)
    original = np.asarray(p['original_cells'])
    if mode == 'fill':
        new_cells, used = _fill(targets, p['palette'], cells, locks, stocks)
    else:
        new_cells, used = _keep(original, cells, locks, stocks)
    p['stocks'] = stocks
    p['mode'] = mode
    p['cells'] = new_cells.tolist()
    record = dict(mode=mode, stocks=list(stocks), ok=True,
                  used=[int(v) for v in used], blanks=int(np.sum(new_cells == BLANK)))
    p['runs'].append(record)
    del p['runs'][:-200]
    return validate_plan(p), record


def failed_run(p, stocks, mode, reason):
    """失败也把这次尝试记入历史，便于追溯库存与方式选择；格内容保持不变。"""
    k = len(p['palette'])
    p['runs'].append(dict(mode=mode_value(mode), stocks=stocks_value(stocks, k),
                          ok=False, reason=str(reason)[:300]))
    del p['runs'][:-200]
    return p


def edit_plan(p, body):
    """生成结果后的手工换色/锁定/解锁。锁格以当前指定颜色占料；空格必须先上色才能锁。"""
    mode = body.get('mode', 'paint')
    if mode not in ('paint', 'lock', 'unlock'):
        raise Invalid('编辑模式无效')
    changes = body.get('changes')
    if not isinstance(changes, list) or not 1 <= len(changes) <= 1024:
        raise Invalid('一次编辑需为1至1024格')
    index = integer(body.get('color', 0), '色号', 0, len(p['palette']) - 1)
    coords = []
    for c in changes:
        if not isinstance(c, list) or len(c) != 2:
            raise Invalid('格坐标需为[x,y]')
        coords.append((integer(c[0], '列', 0, p['width'] - 1),
                       integer(c[1], '行', 0, p['height'] - 1)))
    cells = np.asarray(p['cells'])
    locks = np.asarray(p['locks'], dtype=bool)
    stocks = np.asarray(p['stocks'])
    k = len(p['palette'])
    for x, y in coords:
        if mode == 'paint':
            cells[y, x] = index
        elif mode == 'lock':
            if cells[y, x] == BLANK:
                raise Invalid(f'第{y + 1}行第{x + 1}列是空格，必须先上色才能锁定')
            locks[y, x] = True
        else:
            locks[y, x] = False
    if mode == 'lock':
        used = np.bincount(cells[locks], minlength=k)
        over = [i for i in range(k) if used[i] > stocks[i]]
        if over:
            labels = '、'.join(f'{chr(65 + i)}锁格需{int(used[i])}颗/库存{int(stocks[i])}颗' for i in over)
            raise Invalid('锁定后超过库存：' + labels + '；编辑未执行')
    elif mode == 'paint':
        # 手工上色同样受库存约束：全部有豆格的各色用量不得超过库存，保证剩余量不会为负
        filled = cells != BLANK
        used = np.zeros(k, dtype=int)
        np.add.at(used, cells[filled], 1)
        over = [i for i in range(k) if used[i] > stocks[i]]
        if over:
            labels = '、'.join(f'{chr(65 + i)}已用{int(used[i])}颗/库存{int(stocks[i])}颗' for i in over)
            raise Invalid('手工上色后超过库存：' + labels + '；编辑未执行，可先调高库存或改用其他色')
    p['cells'] = cells.tolist()
    p['locks'] = locks.tolist()
    return validate_plan(p)


def plan_summary(p):
    validate_plan(p)
    cells = np.asarray(p['cells'])
    locks = np.asarray(p['locks'], dtype=bool)
    targets = np.asarray(p['targets'], np.float32)
    original = np.asarray(p['original_cells'])
    k = len(p['palette'])
    filled = cells != BLANK
    used = np.zeros(k, dtype=int)
    np.add.at(used, cells[filled], 1)
    colors = []
    rgb = palette_rgb(p['palette'])
    for i, c in enumerate(p['palette']):
        colors.append(dict(index=i, symbol=chr(65 + i), color=c,
                           stock=int(p['stocks'][i]), used=int(used[i]),
                           remain=int(p['stocks'][i] - used[i])))
    render = rgb[np.where(filled, cells, 0)]
    err = float(np.sum((targets[filled] - render[filled]) ** 2)) if np.any(filled) else 0.0
    substituted = int(np.sum(filled & ~locks & (cells != original)))
    return dict(
        total=p['width'] * p['height'],
        filled=int(np.sum(filled)), blanks=int(np.sum(~filled)),
        locked=int(np.sum(locks)),
        colors=colors,
        squared_rgb_error=err,
        mean_squared_rgb_error=err / max(int(np.sum(filled)), 1),
        substituted=substituted,
        last_run=p['runs'][-1] if p['runs'] else None,
    )


def render_plan(p, cell=20, symbols=True, grid=True, marksub=True):
    validate_plan(p)
    cell = integer(cell, '格子像素', 8, 40)
    w, h = p['width'], p['height']
    original = np.asarray(p['original_cells'])
    locks = np.asarray(p['locks'], dtype=bool)
    im = Image.new('RGB', (w * cell, h * cell), 'white')
    d = ImageDraw.Draw(im)
    for y in range(h):
        for x in range(w):
            idx = p['cells'][y][x]
            x0, y0 = x * cell, y * cell
            box = (x0, y0, x0 + cell - 1, y0 + cell - 1)
            if idx == BLANK:
                # 空格：浅灰底加对角叉，明确不是任何色号，也不计物料
                d.rectangle(box, fill='#f2f2f2')
                d.line((x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3), fill='#b9b9b9', width=1)
                d.line((x0 + cell - 3, y0 + 2, x0 + 2, y0 + cell - 3), fill='#b9b9b9', width=1)
            else:
                color = p['palette'][idx]
                d.rectangle(box, fill=color)
                if symbols:
                    rgb1 = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                    ink = '#ffffff' if float(np.mean(rgb1)) < .48 else '#111111'
                    d.text((x0 + cell * .3, y0 + cell * .2), chr(65 + idx), fill=ink)
            # 替代色标记：未锁、有豆且颜色不同于原配色，右上角金色圆点
            if marksub and idx != BLANK and not locks[y, x] and idx != int(original[y, x]):
                r = max(2, cell // 7)
                cx0, cy0 = x0 + cell - r - 3, y0 + 3
                d.ellipse((cx0 - r, cy0 - r, cx0 + r, cy0 + r), fill='#f2b705', outline='#7a5c00')
            if grid:
                d.rectangle(box, outline='#a5a5a5', width=1)
            if p['locks'][y][x]:
                d.rectangle((x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3), outline='#00ccff', width=2)
    out = io.BytesIO()
    im.save(out, format='PNG')
    return out.getvalue()
