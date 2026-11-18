import io, json, zipfile, tempfile, os
import numpy as np
from image_lab import create_app
from beadbench.server import install


def build_client():
    tmp = tempfile.mkdtemp()
    app = create_app(os.path.join(tmp, 'prep'))
    install(app, os.path.join(tmp, 'beads'))
    app.testing = True
    return app.test_client(), tmp


def post(c, path, payload):
    r = c.post(path, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
    return r, r.get_json()


def test_full_plan_lifecycle():
    c, _ = build_client()
    # 演示图 -> 普通图纸 -> 库存排料
    r = c.post('/beads/demo'); assert r.status_code == 200
    img = r.get_json()['image']['id']
    r, j = post(c, '/beads/patterns', dict(image_id=img, width=8, height=8, sampling='area',
                                           palette=['#f4f0df', '#202c46', '#d64d52', '#4e8d68'],
                                           name='集成图纸'))
    assert r.status_code == 200
    pat = j['pattern']; k = len(pat['palette'])

    # 排料：库存总数不足（共8颗 vs 64格），创建应返回 warning 且保留原配色结果
    r, j = post(c, '/beads/plans', dict(pattern_id=pat['id'], stocks=[2, 2, 2, 2], mode='fill',
                                        name='库存排料A'))
    assert r.status_code == 200 and 'warning' in j
    plan = j['plan']
    assert plan['runs'][-1]['ok'] is False
    assert plan['cells'] == pat['cells']  # 上次（初始）有效结果保留

    # 补足库存后排满
    r, j = post(c, '/beads/plans/<key>'.replace('<key>', plan['id']) + '/recompute',
                dict(revision=plan['revision'], stocks=[20, 20, 20, 20], mode='fill'))
    assert r.status_code == 200, j
    plan = j['plan']; summ = j['summary']
    assert summ['blanks'] == 0 and summ['filled'] == 64
    assert all(col['used'] <= col['stock'] for col in summ['colors'])

    # 历史与状态中包含独立的排料条目
    st = c.get('/beads/state').get_json()
    assert any(p['id'] == plan['id'] for p in st['plans'])
    assert any(p['id'] == pat['id'] for p in st['patterns'])
    state_plan = next(p for p in st['plans'] if p['id'] == plan['id'])
    assert 'cells' not in state_plan and state_plan['summary']['filled'] == 64

    # 手工换色 -> 锁定 -> 重算，锁格颜色不变且占料
    r, j = post(c, f"/beads/plans/{plan['id']}/edit",
                dict(revision=plan['revision'], mode='paint', color=1, changes=[[0, 0]]))
    plan = j['plan']
    r, j = post(c, f"/beads/plans/{plan['id']}/edit",
                dict(revision=plan['revision'], mode='lock', changes=[[0, 0]]))
    plan = j['plan']
    assert plan['locks'][0][0] and plan['cells'][0][0] == 1
    r, j = post(c, f"/beads/plans/{plan['id']}/recompute",
                dict(revision=plan['revision'], stocks=[30, 30, 30, 30], mode='fill'))
    assert r.status_code == 200
    plan = j['plan']
    assert plan['cells'][0][0] == 1 and plan['locks'][0][0]

    # 切到保原配色缺料留空：空格出现，且填色=原配色
    r, j = post(c, f"/beads/plans/{plan['id']}/recompute",
                dict(revision=plan['revision'], stocks=[5, 5, 5, 5], mode='keep'))
    assert r.status_code == 200
    plan = j['plan']; summ = j['summary']
    assert summ['blanks'] >= 1
    cells = np.asarray(plan['cells']); orig = np.asarray(plan['original_cells'])
    locks = np.asarray(plan['locks']); f = cells != -1
    # 未锁填色格必须沿用原配色；锁格保留其当前色（可能与原配色不同）
    assert np.all(cells[f & ~locks] == orig[f & ~locks])
    assert cells[0, 0] == 1 and locks[0, 0]  # 锁格颜色未被动
    assert sum(col['used'] for col in summ['colors']) == summ['filled']

    # 空格不能锁：找一个空格尝试锁定，应400且不改 revision 数据
    blank = np.argwhere(cells == -1)[0]
    rev_before = plan['revision']
    r, j = post(c, f"/beads/plans/{plan['id']}/edit",
                dict(revision=rev_before, mode='lock', changes=[[int(blank[1]), int(blank[0])]]))
    assert r.status_code == 400 and '先上色' in j['error']

    # 失败重算返回400但保留上次结果，并把失败尝试持久化
    r, j = post(c, f"/beads/plans/{plan['id']}/recompute",
                dict(revision=rev_before, stocks=[0, 0, 0, 0], mode='keep'))
    assert r.status_code == 400 and '锁格已超过库存' in j['error']
    plan = j['plan']
    assert plan['runs'][-1]['ok'] is False and plan['runs'][-1]['stocks'] == [0, 0, 0, 0]
    # 再取回，失败记录仍在
    again = c.get(f"/beads/plans/{plan['id']}").get_json()['plan']
    assert again['runs'][-1]['ok'] is False

    # PNG 与 ZIP 导出，ZIP 内含库存/方式/锁格/来源
    png = c.get(f"/beads/plans/{plan['id']}.png")
    assert png.status_code == 200 and png.data[:8] == b'\x89PNG\r\n\x1a\n'
    zr = c.get(f"/beads/plans/{plan['id']}.zip")
    assert zr.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(zr.data))
    assert {'plan.png', 'plan.json', 'materials.json', 'README.txt'} <= set(zf.namelist())
    pj = json.loads(zf.read('plan.json'))
    assert pj['stocks'] == plan['stocks'] and pj['mode'] == 'keep'
    assert pj['source']['pattern_id'] == pat['id']
    assert 'original_cells' in pj and 'locks' in pj and 'runs' in pj

    # 普通图纸流程仍可用且不受影响
    r, j = post(c, f"/beads/patterns/{pat['id']}/process",
                dict(revision=pat['revision'], operation='nearest'))
    assert r.status_code == 200
    assert 'stocks' not in j['pattern']


def test_revision_conflict():
    c, _ = build_client()
    c.post('/beads/demo')
    img = c.get('/beads/state').get_json()['images'][-1]['id']
    _, j = post(c, '/beads/patterns', dict(image_id=img, width=6, height=6,
                                           palette=['#111111', '#eeeeee', '#d64d52']))
    pid = j['pattern']['id']
    _, j = post(c, '/beads/plans', dict(pattern_id=pid, stocks=[50, 50, 50], mode='fill'))
    pl = j['plan']
    # 陈旧 revision
    r, j2 = post(c, f"/beads/plans/{pl['id']}/edit",
                 dict(revision=pl['revision'] - 1, mode='unlock', changes=[[0, 0]]))
    assert r.status_code == 409
