#!/usr/bin/env python

import argparse
import json
import os
import queue
import random
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from multiprocessing.managers import BaseManager
from pathlib import Path

BASE = Path(os.environ['EVONAS_WORKDIR'])
SRC = Path(os.environ['EVONAS_SOURCE']) / 'MambaDepthNAS'
PY = os.environ.get('EVONAS_PYTHON', sys.executable)
ROOT = Path(os.environ['EVONAS_RUN_DIR'])
PORT = 50779


def jdump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    os.replace(tmp, path)


def authkey():
    p = ROOT / 'authkey'
    return p.read_bytes()


def container():
    return socket.gethostname()


# ----------------------------------------------------------------------------- manager
class ServerMgr(BaseManager):
    pass


class ClientMgr(BaseManager):
    pass


for _n in ('get_task_q', 'get_result_q', 'get_ctrl_q'):
    ClientMgr.register(_n)


def gpu_apps():
    """{uuid: [pid,...]} of compute processes currently on each GPU (all containers)."""
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    res = {}
    for line in out.strip().splitlines():
        if line.strip():
            u, p = [x.strip() for x in line.split(',')]
            res.setdefault(u, []).append(int(p))
    return res


# ----------------------------------------------------------------------------- worker
class _TQ:
    def __init__(self, q):
        self.q, self.t_get = q, None

    def get(self):
        item = self.q.get()
        self.t_get = time.time()
        return item


class _RQ:
    def __init__(self, q, tq, wid, gpu):
        self.q, self.tq, self.wid, self.gpu = q, tq, wid, gpu

    def put(self, item):
        idx, abs_rel = item
        vals = abs_rel.tolist() if hasattr(abs_rel, 'tolist') else list(abs_rel)
        self.q.put(dict(idx=idx, abs_rel=vals, wid=self.wid, gpu=self.gpu,
                        t_get=self.tq.t_get, t_done=time.time()))


def worker_main(plan_path, wid):
    plan = json.loads(Path(plan_path).read_text())
    w = plan['workers'][wid]
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    mgr = ClientMgr(address=tuple(plan['addr']), authkey=authkey())
    mgr.connect()
    tq, rq, ctrl = mgr.get_task_q(wid), mgr.get_result_q(), mgr.get_ctrl_q()
    ctrl.put(dict(kind='hello', wid=wid, t=time.time(), pid=os.getpid(), host=container()))
    wtq = _TQ(tq)
    wrq = _RQ(rq, wtq, wid, w['gpu'])
    if plan.get('fake'):
        # CPU-only orchestration test: same message protocol, no torch.
        while True:
            task = wtq.get()
            if task is None:
                return
            start, toks = task
            time.sleep(float(plan['fake']) * len(toks) * (1 + 0.3 * (start % 7 == 0)))
            wrq.put((start, [0.1 + 0.001 * (start + i) for i in range(len(toks))]))
    os.chdir(SRC.parent)  # data_splits/ is relative to source/
    sys.path.insert(0, str(SRC))
    sys.argv = ['search.py', plan['args_file']]
    import torch
    import search
    torch.set_num_threads(1)
    import numpy as np
    runtime_seed = int(plan['runtime_seed']) + wid
    random.seed(runtime_seed); np.random.seed(runtime_seed); torch.manual_seed(runtime_seed); torch.cuda.manual_seed_all(runtime_seed)
    print(f'R1-2 runtime_seed={plan["runtime_seed"]} worker_seed={runtime_seed} wid={wid}', flush=True)
    args = search.parse_args()
    args.distributed = False
    args.ckpt_path = plan['ckpt']
    search.worker_main_eval_func(wtq, wrq, args)


# ----------------------------------------------------------------------------- agent
def agent_main():
    me = container()
    (ROOT / 'status').mkdir(parents=True, exist_ok=True)
    gpus = json.loads((ROOT / 'gpus.json').read_text())
    mine = {g['uuid'] for g in gpus if g['container'] == me}
    running, sampler, cond, logs = [], None, None, []

    def status(state, **kw):
        jdump(ROOT / 'status' / f'{me}.json', dict(cond_id=cond, state=state, t=time.time(), **kw))

    def stop_all():
        nonlocal running, sampler, logs
        for p in running:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        t0 = time.time()
        while any(p.poll() is None for p in running) and time.time() - t0 < 60:
            time.sleep(0.5)
        for p in running:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if sampler is not None and sampler.poll() is None:
            sampler.terminate()
        for f in logs:
            f.close()
        running, sampler, logs = [], None, []

    status('idle')
    while True:
        time.sleep(1)
        if (ROOT / 'AGENT_STOP').exists():
            stop_all()
            return
        pf = ROOT / 'plan' / 'current.json'
        if not pf.exists():
            continue
        plan = json.loads(pf.read_text())
        if plan['cond_id'] != cond:
            stop_all()
            cond = plan['cond_id']
            idxs = [i for i, w in enumerate(plan['workers']) if w['container'] == me]
            if not idxs:
                status('idle')
                continue
            used = {plan['workers'][i]['gpu'] for i in idxs}
            apps = gpu_apps()
            foreign = {u: apps[u] for u in used if u in apps}
            if foreign and not plan.get('fake'):
                status('refused', detail=f'foreign compute processes on {foreign}')
                continue
            run_dir = Path(plan['run_dir'])
            run_dir.mkdir(parents=True, exist_ok=True)
            lf = open(run_dir / f'gpu_{me}.csv', 'w')
            logs.append(lf)
            sampler = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,'
                 'temperature.gpu,clocks.sm', '--format=csv,noheader,nounits', '-l', '1'], stdout=lf)
            for i in idxs:
                w = plan['workers'][i]
                f = open(run_dir / f'worker_{i}.log', 'w')
                logs.append(f)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=w['gpu'])
                running.append(subprocess.Popen(
                    [PY, '-B', os.path.abspath(__file__), 'worker', '--plan', str(pf), '--wid', str(i)],
                    stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True))
            status('started', workers=len(running))
        elif running:
            codes = [p.poll() for p in running]
            if any(c not in (None, 0) for c in codes):
                status('failed', codes=codes)
            elif all(c == 0 for c in codes):
                status('stopped')
            else:
                status('running', workers=sum(c is None for c in codes))


# ----------------------------------------------------------------------------- driver
def gen_candidates(n, seed, dataset_args_file):
    """n random architectures from the search's own sampler (the initial DMMPE population)."""
    import numpy as np
    os.chdir(SRC.parent)  # data_splits/ is relative to source/
    sys.path.insert(0, str(SRC))
    sys.argv = ['search.py', dataset_args_file]
    import search
    from search_space import MambaSearchSpace
    args = search.parse_args()
    ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)
    xl = np.zeros(28)
    xu = np.array([len(ss.mlp_ratio) - 1, len(ss.d_state) - 1, len(ss.ssd_expand) - 1] * (ss.num_stages - 1)
                  + [len(ss.mlp_ratio) - 1] + [1] * sum(ss.depth), dtype=np.int32)

    class _P:  # only n_var/bounds are used by the sampler
        n_var = 28

        def bounds(self):
            return xl, xu
    prob = _P()
    np.random.seed(seed)
    random.seed(seed)
    X = search.SafeIntegerRandomSampling(depth=ss.depth)._do(prob, n)
    return [ss.decode(x) for x in X], X.tolist()


def build_conditions(a, gpus):
    """Condition list; every entry is run `repeats` times as fresh independent processes."""
    ds = a.dataset
    conds = []

    def add(name, G, c, mode, mb):
        conds.append(dict(name=name, dataset=ds, G=G, c=c, mode=mode, mb=mb))
    if a.exp in ('scale', 'all'):
        for G in (1, 2, 4, 8):
            add(f'scale_G{G}', G, 1, 'dynamic', 3)
    if a.exp in ('sched', 'all'):
        for G in (4, 8):
            add(f'sched_static_G{G}', G, 1, 'static', 3)
            add(f'sched_dynamic_G{G}', G, 1, 'dynamic', 3)
        for mb in (1, 6):
            add(f'sched_mb{mb}_G8', 8, 1, 'dynamic', mb)
    if a.exp in ('conc', 'all'):
        for c in (1, 2, 3, 4):
            add(f'conc_c{c}_G8', 8, c, 'dynamic', 3)
    seen, out = set(), []
    for c in conds:
        key = (c['G'], c['c'], c['mode'], c['mb'])
        if key not in seen:
            seen.add(key)
            out.append(c)
    out = [c for c in out if c['G'] <= min(len(gpus), a.maxg)]
    return sorted(out, key=lambda c: -c['G'])  # largest G first (stable)


def wait_status(cond_id, containers, want, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok = True
        for c in containers:
            s = json.loads((ROOT / 'status' / f'{c}.json').read_text()) if (ROOT / 'status' / f'{c}.json').exists() else {}
            if s.get('cond_id') == cond_id and s.get('state') in ('refused', 'failed'):
                raise RuntimeError(f'{c}: {s}')
            if not (s.get('cond_id') == cond_id and s.get('state') in want):
                ok = False
        if ok:
            return
        time.sleep(1)
    raise TimeoutError(f'status {want} not reached for {cond_id}')


def driver_main(a):
    gpus = json.loads((ROOT / 'gpus.json').read_text())
    args_file = str(ROOT / 'args' / f'search_{a.dataset}.txt')
    ckpt = a.ckpt
    cand_file = ROOT / f'candidates_{a.dataset}_{a.pop}_{a.seed}.json'
    # regenerated from the seed every time: the workers need the sampler's native types
    cands, X = gen_candidates(a.pop, a.seed, args_file)
    jdump(cand_file, dict(seed=a.seed, pop=a.pop, X=X, configs=cands))
    my_ip = socket.gethostbyname(socket.gethostname())
    conds = build_conditions(a, gpus)
    if a.only:
        conds = [c for c in conds if c['name'] in a.only.split(',')]
    plan_list = [(r, c) for r in range(a.repeats) for c in conds]  # repeat-major: interleaved
    if a.shuffle:
        random.Random(a.seed).shuffle(plan_list)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'plan').mkdir(exist_ok=True)
    state = dict(tqs=[], rq=queue.Queue(), ctrl=queue.Queue(), mode='dynamic')
    ServerMgr.register('get_task_q', callable=lambda wid: state['tqs'][wid if state['mode'] == 'static' else 0])
    ServerMgr.register('get_result_q', callable=lambda: state['rq'])
    ServerMgr.register('get_ctrl_q', callable=lambda: state['ctrl'])
    srv = ServerMgr(address=('0.0.0.0', PORT), authkey=authkey()).get_server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for rep, c in plan_list:
        tag = f"{c['dataset']}_{c['name']}_r{rep}"
        out = ROOT / 'results' / f'{tag}.json'
        if out.exists():
            print('skip', tag, flush=True)
            continue
        W = c['G'] * c['c']
        order = [g for g in gpus][:c['G']]
        workers = [dict(container=order[i // c['c']]['container'], gpu=order[i // c['c']]['uuid'],
                        gpu_index=order[i // c['c']].get('index')) for i in range(W)]
        conts = sorted({w['container'] for w in workers})
        state['mode'] = c['mode']
        state['tqs'] = [queue.Queue() for _ in range(W if c['mode'] == 'static' else 1)]
        state['rq'], state['ctrl'] = queue.Queue(), queue.Queue()
        run_dir = ROOT / 'runs' / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cond_id = f'{tag}_{int(time.time())}'
        plan = dict(cond_id=cond_id, addr=[my_ip, PORT], workers=workers, args_file=args_file,
                    ckpt=ckpt, run_dir=str(run_dir), fake=a.fake, runtime_seed=2026102300+rep)
        rec = dict(tag=tag, rep=rep, cond=c, W=W, workers=workers, pop=a.pop, seed=a.seed,
                   ckpt=ckpt, t_plan=time.time(), rounds=[], status='running', runtime_seed=2026102300+rep, worker_seeds=[2026102300+rep+i for i in range(W)])
        print('RUN', tag, flush=True)
        try:
            (ROOT / 'plan' / 'current.json').unlink(missing_ok=True)
            jdump(ROOT / 'plan' / 'current.json', plan)
            wait_status(cond_id, conts, ('started', 'running'), 300)
            hello = {}
            t0 = time.time()
            while len(hello) < W:
                if time.time() - t0 > 600:
                    raise TimeoutError('workers did not connect')
                try:
                    m = state['ctrl'].get(timeout=2)
                    hello[m['wid']] = m
                except queue.Empty:
                    wait_status(cond_id, conts, ('started', 'running'), 5)
            rec['t_all_connected'] = time.time()
            for rnd in range(a.steady + 1):
                n = len(cands)
                t_start = time.time()
                for s in range(0, n, c['mb']):
                    q = state['tqs'][(s // c['mb']) % W] if c['mode'] == 'static' else state['tqs'][0]
                    q.put((s, cands[s:s + c['mb']]))
                nb = (n + c['mb'] - 1) // c['mb']
                res, got = [None] * n, []
                while len(got) < nb:
                    try:
                        r = state['rq'].get(timeout=5)
                    except queue.Empty:
                        wait_status(cond_id, conts, ('started', 'running'), 5)
                        continue
                    got.append(r)
                    res[r['idx']:r['idx'] + len(r['abs_rel'])] = r['abs_rel']
                t_end = time.time()
                busy = {}
                for r in got:
                    busy.setdefault(r['wid'], 0.0)
                    busy[r['wid']] += r['t_done'] - r['t_get']
                last = {}
                for r in got:
                    last[r['wid']] = max(last.get(r['wid'], 0), r['t_done'])
                wall = t_end - t_start
                rec['rounds'].append(dict(
                    round=rnd, kind='cold' if rnd == 0 else 'steady', wall_s=wall, abs_rel=res,
                    tasks=len(got), busy_s={str(k): v for k, v in sorted(busy.items())},
                    ideal_s=sum(busy.values()) / W,
                    tail_idle_s=statistics.mean(t_end - last.get(i, t_start) for i in range(W)),
                    tasks_per_worker={str(i): sum(1 for r in got if r['wid'] == i) for i in range(W)}))
                print(f"  round {rnd} wall={wall:.1f}s ideal={sum(busy.values()) / W:.1f}s", flush=True)
            rec['status'] = 'ok'
        except Exception as e:  # keep the failure, do not hide it
            rec['status'] = 'failed'
            rec['error'] = f'{type(e).__name__}: {e}'
            print('  FAILED', rec['error'], flush=True)
        finally:
            if c['mode'] == 'static':
                for q in state['tqs']:
                    q.put(None)
            else:
                for _ in range(W):
                    state['tqs'][0].put(None)
            try:
                wait_status(cond_id, conts, ('stopped',), 120)
            except Exception as e:
                rec['stop_note'] = f'{type(e).__name__}: {e}'
            (ROOT / 'plan' / 'current.json').unlink(missing_ok=True)
            time.sleep(3)
            rec['t_end'] = time.time()
            jdump(out if rec['status'] == 'ok' else ROOT / 'results' / f'{tag}.FAILED.json', rec)
        if rec['status'] != 'ok' and not a.keep_going:
            print('stopping after failure', flush=True)
            break
    print('driver done', flush=True)


# ----------------------------------------------------------------------------- summarize
def summarize(a):
    rows = {}
    for f in sorted((ROOT / 'results').glob(f'{a.dataset}_*.json')):
        if f.name.endswith('.FAILED.json'):
            continue
        r = json.loads(f.read_text())
        rows.setdefault(r['cond']['name'], []).append(r)
    ref = None
    print(f"{'condition':22s} n  cold_s       steady_s          ideal_s   eff%   spd_vs_G1")
    base = None
    for name, rs in rows.items():
        steady = [x['wall_s'] for r in rs for x in r['rounds'] if x['kind'] == 'steady']
        cold = [x['wall_s'] for r in rs for x in r['rounds'] if x['kind'] == 'cold']
        ideal = [x['ideal_s'] for r in rs for x in r['rounds'] if x['kind'] == 'steady']
        m = statistics.mean(steady)
        sd = statistics.stdev(steady) if len(steady) > 1 else float('nan')
        if name == 'scale_G1':
            base = m
        W = rs[0]['W']
        spd = (base / m) if base else float('nan')
        eff = 100 * statistics.mean(ideal) / m
        print(f"{name:22s} {len(rs):2d} {statistics.mean(cold):8.1f} {m:9.1f}±{sd:6.1f} {statistics.mean(ideal):9.1f} {eff:6.1f} {spd:7.3f}")
    # numerical consistency across conditions
    allv = [x['abs_rel'] for rs in rows.values() for r in rs for x in r['rounds']]
    if allv:
        ref = allv[0]
        dmax = max(abs(a_ - b_) for v in allv for a_, b_ in zip(v, ref))
        print('max |abs_rel diff| over all rounds/conditions vs first round:', dmax)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='role', required=True)
    sub.add_parser('agent')
    w = sub.add_parser('worker')
    w.add_argument('--plan', required=True)
    w.add_argument('--wid', type=int, required=True)
    d = sub.add_parser('driver')
    d.add_argument('--dataset', choices=['nyu', 'kitti'], default='nyu')
    d.add_argument('--ckpt', required=True)
    d.add_argument('--exp', choices=['scale', 'sched', 'conc', 'all'], default='scale')
    d.add_argument('--pop', type=int, default=96)
    d.add_argument('--seed', type=int, default=1274395)
    d.add_argument('--repeats', type=int, default=3)
    d.add_argument('--steady', type=int, default=2, help='steady rounds after the cold round')
    d.add_argument('--maxg', type=int, default=8, help='skip conditions needing more GPUs')
    d.add_argument('--shuffle', action='store_true')
    d.add_argument('--only', default='', help='comma list of condition names to run')
    d.add_argument('--keep-going', action='store_true')
    d.add_argument('--fake', type=float, default=0, help='CPU-only orchestration test: seconds per model')
    s = sub.add_parser('summarize')
    s.add_argument('--dataset', default='nyu')
    a = ap.parse_args()
    if a.role == 'agent':
        agent_main()
    elif a.role == 'worker':
        worker_main(a.plan, a.wid)
    elif a.role == 'driver':
        driver_main(a)
    else:
        summarize(a)


if __name__ == '__main__':
    main()
