"""Failure evidence and native continuation use offline provider/tool fixtures."""
import asyncio
import base64
import gzip
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from google.genai import types
from test_chia_framework_campaign import no_services as no_services

from ramulator_chia.framework.failures import exception_record, transient_error

pytestmark = pytest.mark.usefixtures('no_services')


def test_nested_errors_keep_leaves_without_credentials():
    try:
        raise httpx.ReadTimeout('Authorization: Bearer SECRET X-CHIA-Token="PRIVATE" api_key=HIDDEN')
    except httpx.ReadTimeout as exc:
        error = ExceptionGroup('task failure', [ExceptionGroup('inner', [exc])])
    record = exception_record(error)
    raw = json.dumps(record)
    assert all(value not in raw for value in ('SECRET', 'PRIVATE', 'HIDDEN'))
    leaf = record['exceptions'][0]['exceptions'][0]
    assert leaf['type'] == 'ReadTimeout' and leaf['traceback']
    assert transient_error(error)
    assert not transient_error(ExceptionGroup('mixed', [error, ValueError('bad config')]))
    assert not transient_error(httpx.LocalProtocolError('invalid request'))
    assert not transient_error(None)


def test_token_count_failure_is_recorded_without_changing_history():
    from ramulator_chia.framework.compaction import AdkCompactor
    from ramulator_chia.framework.config import ContextCompaction
    events = []
    values = [types.Content(role='user', parts=[types.Part(text='preserved history')])]

    def count(**kwargs):
        raise httpx.ConnectError('fixture unavailable')

    policy = ContextCompaction(adk_python='/unused/python', input_limit_tokens=10000,
                               token_counter='vertex_count_tokens')
    model = SimpleNamespace(_pending_tools=False, model='fixture',
                            _native_event=lambda kind, payload: events.append((kind, payload)))
    with pytest.raises(httpx.ConnectError):
        asyncio.run(AdkCompactor(policy, True)(model, SimpleNamespace(models=SimpleNamespace(count_tokens=count)),
                                               values, types.GenerateContentConfig()))
    assert values[0].parts[0].text == 'preserved history'
    assert events[0][0] == 'context_count_error'
    assert events[0][1]['error']['type'] == 'ConnectError'


@pytest.mark.parametrize('failure', ['transient', 'unknown', 'pending_tool'])
def test_reflection_resume_does_not_repeat_completed_write(tmp_path, monkeypatch, failure):
    import mcp
    from test_chia_framework_model_adapters import vertex_fixture, response
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.agent import NativeAgent
    from ramulator_chia.framework.config import VertexBackend
    from ramulator_chia.framework.model_sessions import create_session
    from ramulator_chia.framework.records import TransientFailure

    view = workspace(tmp_path)
    fixture = vertex_fixture(monkeypatch, [
        response(parts=[types.Part(function_call=types.FunctionCall(name='dram__inspect', id='write-1', args={}))]),
        response(text='Summary saved.'),
    ])
    original_call = mcp.ClientSession.call_tool
    writes = []

    async def write_summary(self, name, args):
        writes.append(name)
        view.write('notes/summary.md', 'Preserved reflection, not a new scientific round.')
        return await original_call(self, name, args)

    monkeypatch.setattr(mcp.ClientSession, 'call_tool', write_summary)
    private = tmp_path / 'native-private'
    private.mkdir()

    def create():
        return create_session(VertexBackend(kind='vertex_gemini', model='fixture', reasoning_effort='high',
                                            project='offline', location='global'),
                              session_id='fixture:reflection', private_directory=private,
                              workspace=view.root, system_message='fixture', timeout_seconds=30,
                              vertex_client_kwargs={}, vertex_event_callback=lambda e,s: None)

    @contextmanager
    def tools(view):
        yield [fixture.tool]

    async def fail_after_tool(adapter, client, history, config):
        if fixture.tools:
            adapter._pending_tools = failure == 'pending_tool'
            leaf = ValueError('fixture deterministic error') if failure == 'unknown' else httpx.ReadTimeout('fixture countTokens')
            raise ExceptionGroup('unhandled errors in a TaskGroup', [leaf])

    model = create()
    model.model.context_compactor = fail_after_tool
    agent = NativeAgent(view, model, tools)
    with pytest.raises(RuntimeError) as raised:
        agent.turn('reflect', {'same': 'frozen inputs'})
    assert isinstance(raised.value, TransientFailure) == (failure == 'transient')
    assert len(writes) == len(fixture.calls) == 1
    assert (view.root / 'notes/summary.md').is_file()
    if failure != 'transient':
        return
    # A new harness object restores the native checkpoint, including the
    # completed tool response. Only the remaining terminal response is needed.
    resumed = NativeAgent(view, create(), tools)
    assert resumed.turn('reflect', {'same': 'frozen inputs'})['success']
    assert len(writes) == 1 and len(fixture.calls) == 2
    assert len(fixture.calls[-1]['contents']) == 3
    receipts = [json.loads(p.read_text()) for p in agent.root.glob('reflect-*/receipt.json')]
    assert sorted(r['success'] for r in receipts) == [False, True]
    assert sum(len(r['usage']) for r in receipts) == 2
    failures = [json.loads(gzip.decompress(p.read_bytes())) for p in agent.root.glob('reflect-*/result.json.gz')]
    recorded = next(r for r in failures if not r['success'])
    assert recorded['error']['exceptions'][0]['type'] == 'ReadTimeout'


@pytest.mark.parametrize('pending', [False, True])
def test_sdk_interrupt_preserves_latest_tools_signatures_and_usage(tmp_path, monkeypatch, pending):
    from google import genai
    from test_chia_framework_model_adapters import vertex_fixture, response
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.agent import NativeAgent
    from ramulator_chia.framework.config import VertexBackend
    from ramulator_chia.framework.model_sessions import create_session

    view = workspace(tmp_path)
    fixture = vertex_fixture(monkeypatch, [
        response(parts=[types.Part(
            thought_signature=b'opaque-signature',
            function_call=types.FunctionCall(name='dram__inspect', id='preserve-tool', args={}),
        )]),
        response(text='continued, without repeating the tool'),
    ])
    original_client = genai.Client
    calls = []

    def client(**kwargs):
        value = original_client(**kwargs)
        generate = value.models.generate_content

        def interrupted(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt('fixture operator cancellation')
            return generate(**kwargs)

        value.models.generate_content = interrupted
        return value

    monkeypatch.setattr(genai, 'Client', client)
    private = tmp_path / 'private'
    private.mkdir()

    def create():
        return create_session(VertexBackend(kind='vertex_gemini', model='fixture',
                              reasoning_effort='high', project='offline', location='global'),
                              session_id='fixture:interrupt', private_directory=private,
                              workspace=view.root, system_message='fixture', timeout_seconds=30,
                              vertex_client_kwargs={}, vertex_event_callback=lambda e, s: None)

    @contextmanager
    def tools(view):
        yield [fixture.tool]

    async def mark_ambiguous(adapter, client, history, config):
        if fixture.tools and pending:
            adapter._pending_tools = True

    model = create()
    model.model.context_compactor = mark_ambiguous
    agent = NativeAgent(view, model, tools)
    with pytest.raises(KeyboardInterrupt, match='fixture operator cancellation'):
        agent.turn('explore', {'same': 'inputs'})
    assert agent.active_attempt is None
    assert len(fixture.tools) == 1
    pointer = json.loads((agent.root / 'continuation.json').read_text())
    attempt = agent.root / pointer['directory']
    saved = json.loads(gzip.decompress((attempt / 'state/contents.json.gz').read_bytes()))
    assert saved == json.loads(gzip.decompress((attempt / 'sdk-latest-state.json.gz').read_bytes()))
    assert saved['pending_tools'] is pending
    assert 'preserve-tool' in json.dumps(saved)
    assert 'opaque-signature' == base64.b64decode(
        saved['contents'][1]['parts'][0]['thought_signature']).decode()
    receipt = json.loads((attempt / 'receipt.json').read_text())
    assert receipt['success'] is False and receipt['text'] == ''
    assert [r['status'] for r in receipt['usage']] == [
        'response_recorded', 'awaiting_response_usage_unknown']
    raw = json.loads(gzip.decompress((attempt / 'result.json.gz').read_bytes()))
    assert raw['interrupted'] and raw['error']['type'] == 'KeyboardInterrupt'
    resumed = NativeAgent(view, create(), tools)
    if pending:
        with pytest.raises(RuntimeError):
            resumed.turn('explore', {'same': 'inputs'})
        assert len(calls) == 2  # Ambiguous tools block even an operator restart.
    else:
        assert resumed.turn('explore', {'same': 'inputs'})['success']
        assert len(calls) == 3 and len(fixture.calls[-1]['contents']) == 3
    assert len(fixture.tools) == 1
    assert json.loads((attempt / 'receipt.json').read_text()) == receipt
