import io
from flask import request,jsonify,render_template,send_file
from image_lab.validation import Invalid
from .storage import Storage
from .core import decode,demo,create_pattern,edit_cells,transform,render,summary,png,DEFAULT_PALETTE,integer

def install(app,root):
    store=Storage(root)
    app.extensions['beads']=store
    def body():
        b=request.get_json(silent=True)
        if not isinstance(b,dict):raise Invalid('请求需为JSON对象')
        return b
    def pattern_response(p):return jsonify(pattern=p,summary=summary(p))
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
        if b.get('revision')!=p['revision']:from image_lab.validation import Conflict;raise Conflict('图纸已改变，请刷新后重试')
        q=transform(p,b.get('operation'),b);return pattern_response(store.save_pattern(q,dict(pattern_id=p['id'],pattern_revision=p['revision'],operation=b.get('operation'))))
    @app.get('/beads/patterns/<key>.png')
    def beads_png(key):return send_file(io.BytesIO(render(store.pattern(key),integer(request.args.get('cell',20),'格像素',8,40),request.args.get('symbols','1')=='1',request.args.get('grid','1')=='1')),mimetype='image/png',download_name='beads.png')
    @app.get('/beads/patterns/<key>.zip')
    def beads_zip(key):return send_file(io.BytesIO(store.export(key)),mimetype='application/zip',download_name='beadbench-pattern.zip',as_attachment=True)
    @app.post('/beads/notes')
    def beads_notes():
        b=body();store.notes(b.get('notes',''),b.get('revision'));return jsonify(store.state())
