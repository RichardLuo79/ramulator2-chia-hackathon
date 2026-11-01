"""Single-core O3 ROI screening: 2M warmup, then <=2B instructions.

Uses gem5's instruction probes and benchmark markers. This driver does not
generate checkpoints, change their PCs, replay workloads, or run agents.
An external eight-hour subprocess deadline includes initialization and output.
"""
import argparse
import json
from pathlib import Path
import resource
import re
import sys
import time

from fs_board import IMAGES, loaded_libraries, save_json, check_workload_output
from checkpoint import checkpoint_workload


def window_result(begin, end, target, reason):
    """A partial process/timeout is not a completed observation."""
    actual = end['instructions'] - begin['instructions']
    if actual <= 0 or reason not in ('instruction_cap', 'roi_end'):
        raise ValueError('no valid measured ROI interval')
    if reason == 'instruction_cap' and not target <= actual <= target + 64:
        raise ValueError('instruction limit did not complete at a normal commit boundary')
    return dict(measured_instructions=actual, requested_instructions=target,
                stop_reason=reason, full_roi=False, warmup_prefix_excluded=True,
                simulated_ticks=end['tick']-begin['tick'],
                wall_seconds=end['wall']-begin['wall'],
                cpu_seconds=end['cpu']-begin['cpu'],
                mips=actual/(end['wall']-begin['wall'])/1e6,
                max_rss_kib=end['max_rss_kib'])


def last_statistics(text):
    """gem5 appends dumps; never mix warmup with the measured statistics."""
    marker = '---------- Begin Simulation Statistics ----------'
    if marker not in text:
        raise ValueError('missing statistics dump')
    return marker + text.rsplit(marker, 1)[1]


def main():
    started=time.monotonic()
    import m5
    from m5.objects import GlobalInstTracker, LocalInstTracker
    from gem5.components.boards.x86_board import X86Board
    from gem5.components.boards.simple_board import SimpleBoard
    from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
    from gem5.components.processors.cpu_types import CPUTypes
    from gem5.components.processors.simple_switchable_processor import SimpleSwitchableProcessor
    from gem5.components.processors.simple_processor import SimpleProcessor
    from gem5.isas import ISA
    from gem5.resources.resource import DiskImageResource, KernelResource, BinaryResource
    from gem5.simulate.exit_handler import register_exit_handler
    from gem5.simulate.exit_event import ExitEvent
    from gem5.simulate.simulator import Simulator

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('memory-config','ramulator-python'):
        parser.add_argument('--'+name,required=True,type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint',type=Path)
    group.add_argument('--binary',type=Path)
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--warmup',type=int,default=2_000_000)
    parser.add_argument('--instructions',type=int,default=2_000_000_000)
    args=parser.parse_args()
    if args.warmup<=0 or args.instructions<=0:raise ValueError('instruction limits must be positive')
    sys.path.insert(0,str(args.ramulator_python))
    from ramulator.gem5 import Memory
    config=json.loads(args.memory_config.read_text())
    kind=config['memory_system']['controllers'][0]['impl']
    if kind not in ('GenericDDR','FixedLat','MD1','WMG1','Mess','Atomic'):
        raise ValueError('unknown frozen-model controller')
    if kind == 'Atomic' and not config['memory_system']['controllers'][0].get('model_library'):
        raise ValueError('frozen Atomic model requires an explicit source-bound library')
    output=Path(m5.options.outdir)
    legacy=False

    class Processor(SimpleSwitchableProcessor):
        def incorporate_processor(self,board):
            super().incorporate_processor(board)
            for core in self._all_cores():
                for obj in core.get_simobject().descendants():obj.eventq_index=0
                core.get_simobject().eventq_index=0
        def _pre_instantiate(self,root):
            super()._pre_instantiate(root);root.sim_quantum=0

    if args.checkpoint:
        metadata=checkpoint_workload(args.checkpoint)
        if metadata.get('stage')!='roi' or metadata.get('cores')!=1:
            raise ValueError('a preserved single-core workload ROI checkpoint is required')
        suite=metadata['suite'];legacy=suite=='parsec'
        if suite not in IMAGES or not args.resources:raise ValueError('missing FS image identity')
        processor=Processor(starting_core_type=CPUTypes.O3,switch_core_type=CPUTypes.ATOMIC,
                            isa=ISA.X86,num_cores=1)
        board_type=X86Board
    else:
        metadata={'mode':'SE','binary':str(args.binary),'roi':'explicit workbegin/workend around kernel'}
        processor=SimpleProcessor(cpu_type=CPUTypes.O3,isa=ISA.X86,num_cores=1)
        board_type=SimpleBoard
    board=board_type(clk_freq='3.2GHz',processor=processor,
        memory=Memory(config['memory_system'],size='3GiB'),
        cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(l1d_size='32KiB',l1i_size='32KiB',l2_size='1MiB'))
    board.global_tracker=GlobalInstTracker(inst_thresholds=[])
    processor.get_cores()[0].get_simobject().probeListener=LocalInstTracker(
        global_inst_tracker=board.global_tracker,start_listening=True)
    if args.checkpoint:
        kernel='x86-linux-kernel-4.19.83-1.0.0' if legacy else 'x86-linux-kernel-6.8.0-52-generic-1.0.0'
        partition='1' if legacy else '2'
        device=board.get_disk_device()+'1' if legacy else '/dev/sda2'
        board.set_kernel_disk_workload(kernel=KernelResource(local_path=str(args.resources/kernel)),
            disk_image=DiskImageResource(local_path=str(args.resources/IMAGES[suite]),root_partition=partition),
            kernel_args=['earlyprintk=ttyS0','console=ttyS0','lpj=7999923','root='+device],
            readfile_contents='',checkpoint=args.checkpoint)
    else:
        board.set_se_binary_workload(binary=BinaryResource(local_path=str(args.binary)))
    state={'roi_begin':bool(args.checkpoint),'roi_end':False,'exited':False}
    events=[]

    def record(name):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        value=dict(event=name,tick=int(m5.curTick()),instructions=int(board.global_tracker.getCounter()),
                   wall=time.monotonic(),cpu=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss)
        events.append(value)
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        save_json(output/'progress.json',value)
        return value

    def mark(name):
        if name=='roi_begin' and state['roi_begin']:raise RuntimeError('duplicate ROI begin')
        if name=='roi_end' and (not state['roi_begin'] or state['roi_end']):raise RuntimeError('invalid ROI marker order')
        state[name]=True;record(name)

    def handler(name):
        def callback(simulator,payload,**kwargs):mark(name);return True
        return callback

    def gen(name=None):
        while True:
            if name:mark(name)
            yield True

    for number,name in ((3,'exited'),(4,'roi_begin'),(5,'roi_end')):
        register_exit_handler(number,handler(name),'roi_study_'+name)
    simulator=Simulator(board=board,max_ticks=0,on_exit_event={ExitEvent.MAX_INSTS:gen(),
        ExitEvent.WORKBEGIN:gen('roi_begin'),ExitEvent.WORKEND:gen('roi_end'),ExitEvent.EXIT:gen('exited')})
    simulator.run()
    libraries=loaded_libraries()
    restored=record('restored' if args.checkpoint else 'instantiated')
    simulator.set_max_ticks(10**10)  # Progress only, never a scored cutoff.
    last_report=time.monotonic()

    def advance():
        nonlocal last_report
        simulator.run()
        if time.monotonic()-last_report>30:record('progress');last_report=time.monotonic()
        if state['exited'] and not state['roi_end']:raise RuntimeError('guest exited before ROI end')

    while not state['roi_begin']:advance()
    warmup_begin=record('warmup_begin')
    threshold=warmup_begin['instructions']+args.warmup
    board.global_tracker.addThreshold(threshold)
    while int(board.global_tracker.getCounter())<threshold and not state['roi_end']:advance()
    if state['roi_end']:raise RuntimeError('ROI ended before warmup completed; no score')
    record('warmup_end');m5.stats.dump()
    (output/'warmup.stats.txt').write_text(last_statistics((output/'stats.txt').read_text()))
    m5.stats.reset()  # Retain the predictor, caches, queues and outstanding work.
    begin=record('measurement_begin')
    threshold=begin['instructions']+args.instructions
    board.global_tracker.addThreshold(threshold)
    while int(board.global_tracker.getCounter())<threshold and not state['roi_end']:advance()
    end=record('measurement_end')
    reason='roi_end' if state['roi_end'] else 'instruction_cap'
    m5.stats.dump();(output/'roi.stats.txt').write_text(last_statistics((output/'stats.txt').read_text()))
    result=window_result(begin,end,args.instructions,reason)
    # Capped windows cannot claim benchmark-output verification. For natural
    # ROI ends, execute the ordinary epilogue and retain its output separately.
    verification='unavailable: instruction cap'
    if reason=='roi_end':
        if args.checkpoint:
            # These preserved ROI checkpoints resume older guest scripts.
            # Their completion handshake is writefile + an exit-status line,
            # not a subsequent hypercall. Reuse that exact captured protocol.
            capture=output/('npb-qualification.txt' if suite=='npb' else 'workload-qualification.txt')
            simulator.set_max_ticks(10**9)
            while True:
                if capture.exists():
                    text=capture.read_text(errors='replace')
                    if re.search(r'^QUALIFICATION_EXIT_CODE=-?\d+$',text,re.M):
                        verification=check_workload_output(suite,text,exit_marker='QUALIFICATION_EXIT_CODE')
                        record('guest_output_received');break
                if state['exited']:raise RuntimeError('guest exited without its verification output')
                advance()
        else:
            while not state['exited']:advance()
            verification='SE epilogue completed; result arrays retained'
    final=record('finished')
    result.update(metadata=metadata,events=events,loaded_libraries=libraries,
        adapter='qualified-polling',initialization_seconds=restored['wall']-started,
        setup_seconds=warmup_begin['wall']-restored['wall'],
        warmup_seconds=begin['wall']-warmup_begin['wall'],
        verification_seconds=final['wall']-end['wall'],total_wall_seconds=final['wall']-started,
        verification=verification,
        scope='single-core frozen-model ROI evaluation; no per-request matching')
    save_json(output/'result.json',result)


if __name__ in ('__main__','__m5_main__'):main()
