#!/usr/bin/env python3
"""
Log to PlantUML Converter
Parses API logs with ARCH: markers and generates sequence diagrams
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TraceEvent:
    timestamp: datetime
    component: str
    action: str
    request_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

@dataclass
class RequestTrace:
    request_id: str
    events: List[TraceEvent] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    endpoint: str = ""
    status: str = ""
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

class LogParser:
    # Pattern for ARCH trace markers
    ARCH_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
        r'(?:INFO|DEBUG|WARNING|ERROR)\s+'
        r'ARCH:(\w+)\.(\w+)\s*'
        r'(?:request_id=(\w+))?'
        r'(.*)$'
    )
    
    # Pattern for standard API request logs
    REQUEST_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
        r'(?:INFO)\s+'
        r'(\S+)\s+"(GET|POST|PUT|DELETE|PATCH)\s+([^"]+)"\s+'
        r'(\d+)\s+'
        r'(?:request_id=(\w+))?'
    )
    
    def __init__(self):
        self.traces: Dict[str, RequestTrace] = {}
        
    def parse_line(self, line: str) -> Optional[TraceEvent]:
        """Parse a single log line"""
        # Try ARCH marker first
        match = self.ARCH_PATTERN.match(line.strip())
        if match:
            ts_str, component, action, req_id, extra = match.groups()
            try:
                ts = datetime.fromisoformat(ts_str.replace(' ', 'T'))
            except:
                ts = datetime.now()
            
            req_id = req_id or 'unknown'
            
            # Parse extra key=value pairs
            details = {}
            if extra:
                for kv in re.findall(r'(\w+)=([^\s]+)', extra):
                    details[kv[0]] = kv[1]
                    
            return TraceEvent(
                timestamp=ts,
                component=component,
                action=action,
                request_id=req_id,
                details=details
            )
            
        # Try standard request log
        match = self.REQUEST_PATTERN.match(line.strip())
        if match:
            ts_str, ip, method, path, status, req_id = match.groups()
            try:
                ts = datetime.fromisoformat(ts_str.replace(' ', 'T'))
            except:
                ts = datetime.now()
                
            req_id = req_id or f"req_{ts.timestamp()}"
            
            return TraceEvent(
                timestamp=ts,
                component='router',
                action=f'{method}_{path.split("/")[-1]}',
                request_id=req_id,
                details={'method': method, 'path': path, 'status': status}
            )
            
        return None
        
    def parse_log_file(self, filepath: str):
        """Parse entire log file"""
        with open(filepath, 'r') as f:
            for line in f:
                event = self.parse_line(line)
                if event:
                    if event.request_id not in self.traces:
                        self.traces[event.request_id] = RequestTrace(request_id=event.request_id)
                    
                    trace = self.traces[event.request_id]
                    trace.events.append(event)
                    
                    if not trace.start_time or event.timestamp < trace.start_time:
                        trace.start_time = event.timestamp
                    if not trace.end_time or event.timestamp > trace.end_time:
                        trace.end_time = event.timestamp
                        
    def parse_log_string(self, log_content: str):
        """Parse log content from string"""
        for line in log_content.split('\n'):
            event = self.parse_line(line)
            if event:
                if event.request_id not in self.traces:
                    self.traces[event.request_id] = RequestTrace(request_id=event.request_id)
                
                trace = self.traces[event.request_id]
                trace.events.append(event)
                
                if not trace.start_time or event.timestamp < trace.start_time:
                    trace.start_time = event.timestamp
                if not trace.end_time or event.timestamp > trace.end_time:
                    trace.end_time = event.timestamp

class PlantUMLGenerator:
    # Map components to participant types
    PARTICIPANT_TYPES = {
        'client': 'actor',
        'router': 'participant',
        'pipeline_runner': 'participant', 
        'pipeline': 'participant',
        'job_manager': 'participant',
        'whisper': 'database',
        'audio_analysis': 'participant',
        'diarization': 'participant',
        'database': 'database',
        'db': 'database',
        'sqlite': 'database',
    }
    
    # Component colors by layer
    COLORS = {
        'client': '#4CAF50',
        'router': '#2196F3',
        'pipeline_runner': '#2196F3',
        'pipeline': '#FF9800',
        'job_manager': '#2196F3',
        'whisper': '#FF9800',
        'audio_analysis': '#FF9800',
        'diarization': '#FF9800',
        'database': '#9C27B0',
        'db': '#9C27B0',
        'sqlite': '#9C27B0',
    }
    
    def __init__(self):
        pass
        
    def generate_sequence_diagram(self, trace: RequestTrace, title: str = None) -> str:
        """Generate PlantUML sequence diagram from a request trace"""
        lines = ['@startuml']
        
        # Title
        if title:
            lines.append(f'title {title}')
        else:
            lines.append(f'title Request {trace.request_id}')
            
        lines.append('')
        
        # Skinparam for styling
        lines.append('skinparam participant {')
        lines.append('  BackgroundColor #1a1a2e')
        lines.append('  BorderColor #444')
        lines.append('  FontColor white')
        lines.append('}')
        lines.append('skinparam sequence {')
        lines.append('  ArrowColor #888')
        lines.append('  LifeLineBorderColor #444')
        lines.append('}')
        lines.append('skinparam backgroundColor #12121f')
        lines.append('')
        
        # Collect unique components
        components = set()
        for event in trace.events:
            components.add(event.component)
        
        # Always include client
        components.add('client')
        
        # Declare participants in order
        ordered = ['client'] + sorted([c for c in components if c != 'client'])
        for comp in ordered:
            ptype = self.PARTICIPANT_TYPES.get(comp.lower(), 'participant')
            color = self.COLORS.get(comp.lower(), '#888')
            lines.append(f'{ptype} "{comp}" as {comp} {color}')
            
        lines.append('')
        
        # Generate sequence from events
        prev_component = 'client'
        for i, event in enumerate(sorted(trace.events, key=lambda e: e.timestamp)):
            comp = event.component
            action = event.action
            
            # Format the message
            msg = action.replace('_', ' ').title()
            if event.details:
                detail_str = ', '.join(f'{k}={v}' for k, v in list(event.details.items())[:2])
                if detail_str:
                    msg += f'\\n<size:10>{detail_str}</size>'
            
            # Determine arrow direction
            if i == 0:
                lines.append(f'client -> {comp}: {msg}')
            elif comp != prev_component:
                lines.append(f'{prev_component} -> {comp}: {msg}')
            else:
                lines.append(f'{comp} -> {comp}: {msg}')
                
            prev_component = comp
            
        # Return arrow
        if trace.events:
            status = trace.events[-1].details.get('status', 'OK')
            duration = f'{trace.duration_ms:.0f}ms' if trace.duration_ms else ''
            lines.append(f'{prev_component} --> client: Response {status} {duration}')
            
        lines.append('')
        lines.append('@enduml')
        
        return '\n'.join(lines)
        
    def generate_all_diagrams(self, traces: Dict[str, RequestTrace], output_dir: str):
        """Generate diagram files for all traces"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        manifest = []
        
        for req_id, trace in traces.items():
            if not trace.events:
                continue
                
            puml = self.generate_sequence_diagram(trace)
            filename = f'trace_{req_id}.puml'
            filepath = output_path / filename
            
            with open(filepath, 'w') as f:
                f.write(puml)
                
            manifest.append({
                'request_id': req_id,
                'file': filename,
                'event_count': len(trace.events),
                'duration_ms': trace.duration_ms,
                'start_time': trace.start_time.isoformat() if trace.start_time else None
            })
            
        # Write manifest
        with open(output_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert logs to PlantUML sequence diagrams')
    parser.add_argument('--log', default='/home/melchior/speech3/api/logs/api.log', help='Log file to parse')
    parser.add_argument('--output', default='/home/melchior/speech3/frontend/static/traces', help='Output directory')
    parser.add_argument('--last', type=int, default=10, help='Process last N requests')
    args = parser.parse_args()
    
    print(f"Parsing log: {args.log}")
    log_parser = LogParser()
    log_parser.parse_log_file(args.log)
    
    print(f"Found {len(log_parser.traces)} request traces")
    
    # Take last N
    traces = dict(list(log_parser.traces.items())[-args.last:])
    
    print(f"Generating diagrams for {len(traces)} traces")
    generator = PlantUMLGenerator()
    manifest = generator.generate_all_diagrams(traces, args.output)
    
    print(f"Written {len(manifest)} diagrams to {args.output}")
    
    # Print summary
    for item in manifest:
        print(f"  {item['request_id']}: {item['event_count']} events, {item['duration_ms'] or '?'}ms")

if __name__ == '__main__':
    main()
