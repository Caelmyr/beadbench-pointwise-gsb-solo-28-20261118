import io
from flask import request,jsonify,render_template,send_file
from image_lab.validation import Invalid,Conflict
from .storage import Storage
from .core import decode,demo,create_pattern,edit_cells,transform,render,summary,png,DEFAULT_PALETTE,integer
from .plan import create_plan,run_plan,failed_run,edit_plan,plan_summary,render_plan,stocks_value,mode_value

def install(app,root):
    store=Storage(root)
    app.extensions['beads']=store
    def body():
        b=request.get_json(silent=True)
        if not isinstance(b,dict):raise Invalid('请求需为JSON对象')
        return b
    def pattern_response(p):return jsonify(pattern=p,summary=summary(p))
    def plan_response(p):return jsonify(plan=p,summary=plan_summary(p))
    @app.get('/beads/state')
    def beads_state():return jsonify(store.state())
    @app.get('/')
    @app.get('/beads')
    def beads_page():return render_template('beads.html',palette=DEFAULT_PALETTE)
    @app.post('/beads/demo')
    def beads_demo():return jsonify(image=store.save_image('合成微笑花朵',demo()))
    @app.post('/beads/upload')
    def beads_upload():
        f=request.files.get('file')
        if not f:raise Invalid('请选择照片')
        return jsonify(image=store.save_image(f.filename,decode(f.read())))
    @app.get('/beads/images/<key>.png')
    def beads_image(key):return send_file(io.BytesIO(png(store.image(key)[1])),mimetype='image/png')
    @app.post('/beads/patterns')
    def beads_create():
        b=body();row,a=store.image(b.get('image_id'));return pattern_response(store.save_pattern(create_pattern(a,b),dict(image_id=row['id'],image_sha256=row['sha256'])))
    @app.get('/beads/patterns/<key>')
    def beads_pattern(key):return pattern_response(store.pattern(key))
    @app.post('/beads/patterns/<key>/edit')
    def beads_edit(key):
        b=body();return pattern_response(store.change(key,b.get('revision'),lambda p:edit_cells(p,b)))
    @app.post('/beads/patterns/<key>/notes')
    def beads_pattern_notes(key):
        b=body()
        def update(p):p['notes']=str(b.get('notes',''))[:10000]
        return pattern_response(store.change(key,b.get('revision'),update))
    @app.post('/beads/patterns/<key>/process')
    def beads_process(key):
        b=body();p=store.pattern(key)
        if b.get('revision')!=p['revision']:raise Conflict('图纸已改变，请刷新后重试')
        q=transform(p,b.get('operation'),b);return pattern_response(store.save_pattern(q,dict(pattern_id=p['id'],pattern_revision=p['revision'],operation=b.get('operation'))))
    @app.get('/beads/patterns/<key>.png')
    def beads_png(key):return send_file(io.BytesIO(render(store.pattern(key),integer(request.args.get('cell',20),'格像素',8,40),request.args.get('symbols','1')=='1',request.args.get('grid','1')=='1')),mimetype='image/png',download_name='beads.png')
    @app.get('/beads/patterns/<key>.zip')
    def beads_zip(key):return send_file(io.BytesIO(store.export(key)),mimetype='application/zip',download_name='beadbench-pattern.zip',as_attachment=True)

    # ---- 库存排料（独立历史，不影响普通图纸） ----
    @app.post('/beads/plans')
    def beads_plan_create():
        b=body();p=store.pattern(b.get('pattern_id'))
        pl=create_plan(p,b)
        stocks=pl['stocks'];mode=pl['mode']
        try:
            run_plan(pl,stocks,mode)
        except Invalid as e:
            # 首次排料不可行：记录失败尝试，保留初始（=原配色）结果并明确原因
            failed_run(pl,stocks,mode,e)
            saved=store.save_plan(pl,dict(pattern_id=p['id'],pattern_revision=p['revision'],created=True))
            return jsonify(plan=saved,summary=plan_summary(saved),warning=str(e))
        saved=store.save_plan(pl,dict(pattern_id=p['id'],pattern_revision=p['revision'],created=True))
        return plan_response(saved)
    @app.get('/beads/plans/<key>')
    def beads_plan_get(key):return plan_response(store.plan(key))
    @app.post('/beads/plans/<key>/recompute')
    def beads_plan_recompute(key):
        b=body()
        def work(pl):
            k=len(pl['palette'])
            st=stocks_value(b.get('stocks'),k);md=mode_value(b.get('mode'))
            try:
                run_plan(pl,st,md)
            except Invalid as e:
                failed_run(pl,st,md,e)
                raise Invalid(str(e))
        try:
            saved,error=store.recompute_plan(key,b.get('revision'),work)
        except Conflict:
            raise
        if error:
            return jsonify(error=error,plan=saved,summary=plan_summary(saved)),400
        return plan_response(saved)
    @app.post('/beads/plans/<key>/edit')
    def beads_plan_edit(key):
        b=body()
        saved,_=store.change_plan(key,b.get('revision'),lambda pl:edit_plan(pl,b))
        return plan_response(saved)
    @app.post('/beads/plans/<key>/notes')
    def beads_plan_notes(key):
        b=body()
        def update(pl):pl['notes']=str(b.get('notes',''))[:10000]
        saved,_=store.change_plan(key,b.get('revision'),update)
        return plan_response(saved)
    @app.get('/beads/plans/<key>.png')
    def beads_plan_png(key):
        return send_file(io.BytesIO(render_plan(store.plan(key),integer(request.args.get('cell',20),'格像素',8,40),request.args.get('symbols','1')=='1',request.args.get('grid','1')=='1',request.args.get('marksub','1')=='1')),mimetype='image/png',download_name='beadbench-plan.png')
    @app.get('/beads/plans/<key>.zip')
    def beads_plan_zip(key):return send_file(io.BytesIO(store.export_plan(key)),mimetype='application/zip',download_name='beadbench-material-plan.zip',as_attachment=True)

    @app.post('/beads/notes')
    def beads_notes():
        b=body();store.notes(b.get('notes',''),b.get('revision'));return jsonify(store.state())
