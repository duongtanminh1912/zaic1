"""Linux CUDA candidate. Uses mounted BTC SDK, never a baked TRAIN SDK."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import platform
import site
import sys
import time

CODE=Path(__file__).resolve().parent
REPO=next((p for p in [CODE,*CODE.parents] if (p/'ops/src').is_dir()),CODE)
sys.path[:0]=[str(REPO/'ops/src'),str(REPO/'llm/baseline_52_release/inference')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
from lifemate.runtime.configuration import strict_json
from lifemate.runtime.container_io import split_root,input_rows,timed_submission,check_timed_output
from lifemate.runtime.smoke_inputs import digest


def memory_snapshot():
    """Small read-only Linux/WSL counters; no GPU synchronization or subprocess."""
    result={}
    for filename,keys in [('/proc/meminfo',{'MemAvailable','SwapTotal','SwapFree'}),
                          ('/proc/vmstat',{'pswpin','pswpout'})]:
        try:
            for line in Path(filename).read_text().splitlines():
                parts=line.replace(':','').split()
                if parts and parts[0] in keys:result[parts[0]]=int(parts[1])
        except (OSError,ValueError,IndexError):pass
    for name in ('memory.current','memory.max','memory.swap.current','memory.events'):
        try:result['cgroup_'+name]=Path('/sys/fs/cgroup',name).read_text().strip()
        except OSError:pass
    return result


class StageTimes:
    def __init__(self):self.milliseconds={};self.failure_phase=None

    @contextmanager
    def measure(self,name):
        started=time.perf_counter()
        try:yield
        except BaseException:
            self.failure_phase=self.failure_phase or name
            raise
        finally:self.milliseconds[name]=int((time.perf_counter()-started)*1000)


class Predictor:
    """Large 52 serial placement, with one mounted SDK session per sample."""
    def __init__(self,data_root='/data',assets='/code/private_assets',runtime='/code/linux_runtime',result='/result',split=None,run_label='predict'):
        from baseline50 import actor
        from baseline50.asr import SpeechASR
        from baseline50.backend import LocalServer
        from baseline50.release import verify
        verify()
        started=time.perf_counter()
        if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
            raise ValueError('Candidate requires Linux AMD64/CUDA')
        self.root,self.split,self.directory=split_root(data_root,split)
        self.result=Path(result);self.result.mkdir(parents=True,exist_ok=True)
        self.run_label=run_label
        self.assets=Path(assets);runtime=Path(runtime)
        self.spec=strict_json((REPO/'llm/baseline_52_release/inference/assets.manifest.json').read_text())
        for relative,info in self.spec['files'].items():
            if relative.startswith('models/') or relative in ('private_data/split.json','private_data/retrieval_index.json'):
                p=(self.assets/relative).resolve()
                if not p.is_relative_to(self.assets.resolve()) or p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
                    raise ValueError('Model/private assets differ from frozen Large 52 release')
        receipt=strict_json((runtime/'runtime_manifest.json').read_text())
        if (receipt['release']!='b11800' or receipt['platform']!='linux-amd64-cuda12.8'
                or receipt['server']!='llama-server' or 'llama-server' not in receipt['files']
                or receipt['archive_sha256']!='e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7'):
            raise ValueError('Wrong Linux runtime variant')
        for relative,info in receipt['files'].items():
            p=(runtime/relative).resolve()
            if not p.is_relative_to(runtime.resolve()) or p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
                raise ValueError('Linux runtime failed checksum')
        self.executable=runtime/receipt['server']
        with self.executable.open('rb') as stream:
            if stream.read(4)!=b'\x7fELF':raise ValueError('Expected Linux ELF server')
        if any(n=='zaic_tools' or n.startswith('zaic_tools.') for n in sys.modules):
            raise ValueError('Fresh process required for mounted SDK')
        sys.path.insert(0,str(self.directory));self.sdk=importlib.import_module('zaic_tools')
        if not Path(self.sdk.__file__).resolve().is_relative_to((self.directory/'zaic_tools').resolve()):
            raise ValueError('SDK must come from mounted split')
        catalog=strict_json((self.directory/(self.split+'_tools_catalog.json')).read_text())
        self.tools={t['name']:t for app in catalog['apps'] for t in app['tools']}
        actor.bind_package(self.assets/'private_data')
        self.actor=actor;self.server_factory=LocalServer
        library_dirs=[str(p) for base in site.getsitepackages() for p in Path(base).glob('nvidia/*/lib')]
        os.environ['LD_LIBRARY_PATH']=':'.join([str(runtime.resolve()),*library_dirs,os.environ.get('LD_LIBRARY_PATH','')])
        self.asr=SpeechASR(self.assets/'models/asr',self.root,self.split,self.spec['asr_vocab_sha256'],
                           parameters=1543304960,batch_size=2)
        self.ordinal=0
        print(json.dumps({'model_load_ms':int((time.perf_counter()-started)*1000),
                          'baseline':'52','placement':'serial','warmup':False,'gpu_validated':False}),flush=True)

    def infer(self,prepared,started=None):
        from baseline50.asr import audio_paths,audio_view
        started=time.perf_counter() if started is None else started
        deadline=time.monotonic()+max(0.,30-(time.perf_counter()-started))
        stages=StageTimes()
        input_prepare_ms=int((time.perf_counter()-started)*1000)
        memory_before=memory_snapshot();memory_before_qwen=None;qwen_budget_ms=None
        self.ordinal+=1
        env=self.sdk.ToolEnv(self.split,prepared['id'],root=self.root)
        row={'id':prepared['id'],'audio':prepared['latest_audio'],
             'context':prepared['context'],'history':prepared['history']}
        status='runtime_error';error_type=None;repairs=0
        try:
            if time.monotonic()>=deadline:
                stages.failure_phase='input_preparation'
                raise TimeoutError('Input preparation exhausted deadline')
            with stages.measure('asr_to_gpu_ms'):self.asr.model.to('cuda')
            with stages.measure('asr_transcribe_ms'):
                mapping,_=self.asr.transcribe(audio_paths(row),deadline)
            with stages.measure('asr_to_cpu_ms'):
                self.asr.model.to('cpu');self.asr.torch.cuda.empty_cache()
            if time.monotonic()>=deadline:
                stages.failure_phase='after_asr_transfer'
                raise TimeoutError('ASR/transfer exhausted sample deadline')
            memory_before_qwen=memory_snapshot()
            qwen_budget_ms=max(0,int((deadline-time.monotonic())*1000))
            with stages.measure('qwen_startup_ms'):
                server=self.server_factory(self.assets/self.spec['qwen_file'],self.executable,
                    self.result/(self.run_label+'_server_'+str(self.ordinal)),deadline=deadline)
                backend=server.__enter__()
            try:
                with stages.measure('agent_ms'):
                    result=self.actor.infer(audio_view(row,mapping),self.tools,env,backend,
                        seconds=max(0.,deadline-time.monotonic()),max_calls=8,max_repairs=2)
                    status=result['status'];repairs=result['repairs']
                    if status not in ('finished','asked_missing_info','call_budget_exhausted'):
                        stages.failure_phase='agent'
            finally:
                with stages.measure('qwen_shutdown_ms'):server.__exit__(*sys.exc_info())
        except Exception as exc:
            status='timeout' if isinstance(exc,TimeoutError) else 'runtime_error'
            error_type=type(exc).__name__
            # Save the technical traceback privately; never replace session output.
            import traceback
            with (self.result/(self.run_label+'_errors.log')).open('a') as stream:
                traceback.print_exc(file=stream)
        finally:
            with stages.measure('asr_cleanup_ms'):
                self.asr.model.to('cpu');self.asr.torch.cuda.empty_cache()
        with stages.measure('session_export_ms'):
            submission={'id':prepared['id'],'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}
        memory_after=memory_snapshot()
        ms=int((time.perf_counter()-started)*1000)
        if ms>30000:
            status='timeout';stages.failure_phase=stages.failure_phase or 'end_to_end'
        diagnostics={'id':prepared['id'],'status':status,'error_type':error_type,
                     'failure':status not in ('finished','asked_missing_info','call_budget_exhausted'),
                     'repairs':repairs,'baseline':'52','placement':'serial','warmup':False,
                     'runtime_parity_verified':False,'predict_ms':ms,
                     'stage_ms':stages.milliseconds,'input_prepare_ms':input_prepare_ms,
                     'failure_phase':stages.failure_phase,'qwen_budget_ms':qwen_budget_ms,
                     'startup_failure_includes_backend_cleanup':stages.failure_phase=='qwen_startup_ms',
                     'memory_before':memory_before,'memory_before_qwen':memory_before_qwen,
                     'memory_after':memory_after}
        print(json.dumps(diagnostics),flush=True)
        return timed_submission(submission,ms),diagnostics

    def close(self):
        self.asr.model.to('cpu');self.asr.torch.cuda.empty_cache()


def write_predictions(predictor,filename='submission.jsonl'):
    output=predictor.result/filename
    if output.exists():raise FileExistsError('Preserve existing submission; no automatic rerun')
    ids=[];failures=0;timeouts=0
    with output.open('x',encoding='utf-8') as stream, (predictor.result/(filename+'.diagnostics.jsonl')).open('x',encoding='utf-8') as log:
        iterator=iter(input_rows(predictor.directory,predictor.split))
        while True:
            started=time.perf_counter()
            try:prepared=next(iterator)
            except StopIteration:break
            record,diagnostics=predictor.infer(prepared,started=started)
            failures+=bool(diagnostics['failure']);timeouts+=diagnostics['status']=='timeout'
            ids.append(prepared['id'])
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            log.write(json.dumps(diagnostics,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
    report={**check_timed_output(output,ids),'technical_failures':failures,'timeouts':timeouts,
            'technical_smoke_passed':failures==0,'gpu_timing_scope':'local GPU only'}
    with (predictor.result/(filename+'.summary.json')).open('x',encoding='utf-8') as summary:
        json.dump(report,summary,indent=2)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/data');p.add_argument('--output-dir',default='/result')
    p.add_argument('--assets',default='/code/private_assets');p.add_argument('--runtime',default='/code/linux_runtime')
    p.add_argument('--split');p.add_argument('--check-input',action='store_true')
    args=p.parse_args()
    if args.check_input:
        _,split,directory=split_root(args.data_root,args.split)
        rows=list(input_rows(directory,split))
        print(json.dumps({'input_valid':True,'split':split,'samples':len(rows),'model_inference_run':False}))
        return
    if Path(args.output_dir,'submission.jsonl').exists():raise FileExistsError('Existing output')
    predictor=Predictor(args.data_root,args.assets,args.runtime,args.output_dir,args.split)
    try:
        report=write_predictions(predictor)
        print(json.dumps(report))
    finally:predictor.close()
    if report['technical_failures']:raise SystemExit(2)


if __name__=='__main__':main()
