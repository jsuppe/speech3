#!/usr/bin/env python3
"""
Architecture Generator
Parses @arch annotations from Python and Dart files to generate architecture-data.json
for the 3D architecture visualization.
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Layer colors (matching the 3D visualization)
LAYER_COLORS = {
    'app': '#4CAF50',
    'api': '#2196F3', 
    'pipeline': '#FF9800',
    'data': '#9C27B0'
}

# Layer Y positions in 3D space
LAYER_Y = {
    'app': 15,
    'api': 5,
    'pipeline': -5,
    'data': -15
}

@dataclass
class Region:
    name: str
    desc: str = ""
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 10
    style: str = ""
    deps: List[str] = field(default_factory=list)

@dataclass
class FlowStep:
    order: int
    name: str
    desc: str
    component: str = ""

@dataclass 
class Component:
    name: str
    layer: str
    type: str = "component"
    desc: str = ""
    depends: List[str] = field(default_factory=list)
    external: List[str] = field(default_factory=list)
    regions: List[Region] = field(default_factory=list)
    file_path: str = ""
    size: int = 10  # For 3D box sizing

@dataclass
class Flow:
    name: str
    steps: List[FlowStep] = field(default_factory=list)

class ArchitectureParser:
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.flows: Dict[str, Flow] = {}
        
    def parse_python_file(self, filepath: str):
        """Parse a Python file for @arch annotations and AST info"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except:
            return
            
        # Get file size for component sizing
        file_size = len(content) // 100  # Rough KB
            
        # Extract @arch annotations
        self._parse_arch_annotations(content, filepath, 'python', file_size)
        
        # AST parsing for imports (auto-detect external deps)
        try:
            tree = ast.parse(content)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            self._current_imports = list(set(imports))
        except:
            self._current_imports = []
            
    def parse_dart_file(self, filepath: str):
        """Parse a Dart file for @arch annotations"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except:
            return
            
        file_size = len(content) // 100
        self._parse_arch_annotations(content, filepath, 'dart', file_size)
        
    def _parse_arch_annotations(self, content: str, filepath: str, lang: str, file_size: int = 10):
        """Extract @arch annotations from content"""
        lines = content.split('\n')
        current_component = None
        current_flow = None
        
        for i, line in enumerate(lines):
            # Component declaration
            match = re.search(r'@arch\s+component=(\w+)\s+layer=(\w+)(?:\s+type=(\w+))?', line)
            if match:
                name, layer, comp_type = match.groups()
                comp_type = comp_type or 'component'
                current_component = Component(
                    name=name, 
                    layer=layer, 
                    type=comp_type,
                    file_path=filepath,
                    size=max(5, min(50, file_size))  # Clamp size
                )
                self.components[name] = current_component
                continue
                
            # Dependencies
            match = re.search(r'@arch\s+depends=\[([^\]]+)\]', line)
            if match and current_component:
                deps = [d.strip() for d in match.group(1).split(',')]
                current_component.depends = deps
                continue
                
            # External dependencies
            match = re.search(r'@arch\s+external=\[([^\]]+)\]', line)
            if match and current_component:
                ext = [d.strip() for d in match.group(1).split(',')]
                current_component.external = ext
                continue
                
            # Description
            match = re.search(r'@arch\s+description="([^"]+)"', line)
            if match and current_component:
                current_component.desc = match.group(1)
                continue
                
            # Flow declaration
            match = re.search(r'@arch\.flow\s+(\w+)', line)
            if match:
                flow_name = match.group(1)
                current_flow = Flow(name=flow_name)
                self.flows[flow_name] = current_flow
                continue
                
            # Flow step
            match = re.search(r'@arch\.step\s+(\d+):\s*(\w+)\s+"([^"]+)"', line)
            if match and current_flow:
                order, name, desc = match.groups()
                step = FlowStep(order=int(order), name=name, desc=desc)
                current_flow.steps.append(step)
                continue
                
            # UI Region (Dart)
            match = re.search(r'@arch\.region\s+(\w+)\s*\{([^}]+)\}', line)
            if match and current_component:
                region_name = match.group(1)
                props_str = match.group(2)
                props = {}
                for prop in props_str.split(','):
                    if ':' in prop:
                        k, v = prop.split(':', 1)
                        k = k.strip()
                        v = v.strip()
                        if v.isdigit():
                            props[k] = int(v)
                        else:
                            props[k] = v
                region = Region(name=region_name, **props)
                current_component.regions.append(region)
                continue
                
            # Region description
            match = re.search(r'@arch\.region\.desc\s+"([^"]+)"', line)
            if match and current_component and current_component.regions:
                current_component.regions[-1].desc = match.group(1)
                
    def parse_directory(self, dirpath: str, extensions: List[str] = None):
        """Recursively parse all matching files in directory"""
        extensions = extensions or ['.py', '.dart']
        
        for root, dirs, files in os.walk(dirpath):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'build', '__pycache__', 'node_modules', '.dart_tool']]
            
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                
                if ext == '.py' and '.py' in extensions:
                    self.parse_python_file(filepath)
                elif ext == '.dart' and '.dart' in extensions:
                    self.parse_dart_file(filepath)
                    
    def _build_dependency_graph(self) -> Dict[str, Dict[str, List[str]]]:
        """Build dependency graph from components"""
        graph = {}
        for name, comp in self.components.items():
            graph[name] = {
                'internal': comp.depends,
                'external': comp.external
            }
        return graph
        
    def _build_layers(self) -> Dict[str, Any]:
        """Build layer structure for 3D visualization"""
        layers = {
            'app': {'name': 'Flutter App', 'color': LAYER_COLORS['app'], 'y': LAYER_Y['app'], 'components': []},
            'api': {'name': 'FastAPI Backend', 'color': LAYER_COLORS['api'], 'y': LAYER_Y['api'], 'components': []},
            'pipeline': {'name': 'Analysis Pipeline', 'color': LAYER_COLORS['pipeline'], 'y': LAYER_Y['pipeline'], 'components': []},
            'data': {'name': 'Data Layer', 'color': LAYER_COLORS['data'], 'y': LAYER_Y['data'], 'components': []},
        }
        
        for name, comp in self.components.items():
            layer = comp.layer
            if layer in layers:
                layers[layer]['components'].append({
                    'name': name,
                    'size': comp.size,
                    'desc': comp.desc,
                    'type': comp.type
                })
                
        return layers
        
    def _build_component_details(self) -> Dict[str, Any]:
        """Build component details for drill-down views"""
        details = {}
        
        for name, comp in self.components.items():
            color = LAYER_COLORS.get(comp.layer, '#888888')
            
            # Determine visualization type
            if comp.type == 'screen':
                vis_type = 'phone'
            elif comp.type in ['service', 'router', 'module']:
                vis_type = 'flow'
            elif comp.type == 'database':
                vis_type = 'schema'
            else:
                vis_type = 'flow'
                
            detail = {
                'color': color,
                'type': vis_type,
                'desc': comp.desc,
                'depends': comp.depends,
                'external': comp.external,
                'file_path': comp.file_path
            }
            
            # Add regions for phone type
            if vis_type == 'phone' and comp.regions:
                detail['regions'] = [asdict(r) for r in comp.regions]
            # Add schema details for database type
            elif vis_type == 'schema' and name in DATABASE_SCHEMA:
                detail['tables'] = DATABASE_SCHEMA[name]['tables']
                detail['relations'] = DATABASE_SCHEMA[name]['relations']
            # Add nodes for flow type
            elif vis_type == 'flow':
                # Auto-generate nodes from dependencies
                nodes = []
                y_pos = 20
                for i, dep in enumerate(comp.depends[:6]):  # Max 6 nodes
                    nodes.append({
                        'name': dep,
                        'desc': f'Dependency: {dep}',
                        'x': 30 + (i % 2) * 40,
                        'y': y_pos + (i // 2) * 30,
                        'deps': []
                    })
                if nodes:
                    detail['nodes'] = nodes
                    detail['connections'] = []
                    
            details[name] = detail
            
        return details
                    
    def to_json(self) -> str:
        """Export architecture data as JSON for the 3D visualization"""
        data = {
            "architecture": self._build_layers(),
            "componentDetails": self._build_component_details(),
            "dependencyGraph": self._build_dependency_graph(),
            "flows": {},
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "component_count": len(self.components),
                "flow_count": len(self.flows)
            }
        }
        
        for name, flow in self.flows.items():
            data["flows"][name] = {
                "steps": [asdict(s) for s in sorted(flow.steps, key=lambda x: x.order)]
            }
            
        return json.dumps(data, indent=2)

def add_default_components(parser: ArchitectureParser):
    """Add default/fallback components for items not yet annotated"""
    defaults = [
        # App layer
        Component("HomeScreen", "app", "screen", "Main dashboard", ["APIService"], [], [], "", 16),
        Component("RecordScreen", "app", "screen", "Audio recording UI", ["APIService"], [], [], "", 21),
        Component("CoachScreen", "app", "screen", "AI speech coach", ["APIService"], [], [], "", 10),
        Component("HistoryScreen", "app", "screen", "Past recordings", ["APIService"], [], [], "", 16),
        Component("ProjectsScreen", "app", "screen", "Project management", ["APIService"], [], [], "", 13),
        Component("SettingsScreen", "app", "screen", "User preferences", ["APIService", "AuthService"], [], [], "", 21),
        Component("APIService", "app", "service", "Backend communication", ["AuthService"], ["dio"], [], "", 16),
        Component("AuthService", "app", "service", "Google/Apple sign-in", [], ["google_sign_in"], [], "", 8),
        # API layer
        Component("main.py", "api", "module", "App entry, lifespan", ["router"], ["fastapi", "uvicorn"], [], "", 8),
        Component("pipeline_runner", "api", "module", "Analysis orchestration", ["Whisper STT", "audio_analysis"], ["faster_whisper"], [], "", 28),
        Component("job_manager", "api", "module", "Async job queue", ["pipeline_runner"], [], [], "", 15),
        Component("auth", "api", "module", "API key auth", [], ["fastapi"], [], "", 9),
        Component("user_auth", "api", "module", "JWT/OAuth", [], ["python-jose"], [], "", 12),
        # Pipeline layer
        Component("Whisper STT", "pipeline", "module", "Speech-to-text (GPU)", [], ["faster_whisper", "CUDA"], [], "", 20),
        Component("audio_analysis", "pipeline", "module", "Pitch, jitter, shimmer", [], ["parselmouth"], [], "", 8),
        Component("audio_quality", "pipeline", "module", "SNR, clipping detection", [], ["numpy"], [], "", 4),
        Component("text_analysis", "pipeline", "module", "Complexity, fluency", [], ["spacy"], [], "", 4),
        Component("sentiment", "pipeline", "module", "Emotion detection", [], ["nltk"], [], "", 3),
        Component("diarization", "pipeline", "module", "Speaker identification", [], ["resemblyzer"], [], "", 16),
        # Data layer
        Component("SQLite DB", "data", "database", "11GB, 77K+ speeches", [], ["sqlite3"], [], "", 52),
        Component("speeches", "data", "table", "Recording metadata", ["SQLite DB"], [], [], "", 20),
        Component("analyses", "data", "table", "Full analysis JSON", ["SQLite DB"], [], [], "", 15),
        Component("users", "data", "table", "User accounts", ["SQLite DB"], [], [], "", 10),
        Component("projects", "data", "table", "Project groupings", ["SQLite DB"], [], [], "", 8),
        Component("audio_files", "data", "table", "Opus audio storage", ["SQLite DB"], [], [], "", 30),
    ]
    
    for comp in defaults:
        if comp.name not in parser.components:
            parser.components[comp.name] = comp

# Rich schema data for database visualization
DATABASE_SCHEMA = {
    "SQLite DB": {
        "tables": [
            { "name": "speeches", "desc": "Recording metadata (77K+ rows)", "cols": ["id", "user_id", "title", "duration", "created_at"] },
            { "name": "analyses", "desc": "Full JSON results", "cols": ["speech_id", "results_json", "score"] },
            { "name": "users", "desc": "User accounts", "cols": ["id", "email", "name", "created_at"] },
            { "name": "speaker_embeddings", "desc": "Voice fingerprints", "cols": ["user_id", "name", "embedding[256]"] },
            { "name": "projects", "desc": "Recording groupings", "cols": ["id", "user_id", "name", "type"] },
            { "name": "coach_messages", "desc": "AI chat history", "cols": ["user_id", "speech_id", "role", "content"] }
        ],
        "relations": [
            { "from": 0, "to": 1, "label": "1:1" },
            { "from": 2, "to": 0, "label": "1:N" },
            { "from": 2, "to": 3, "label": "1:N" },
            { "from": 2, "to": 4, "label": "1:N" },
            { "from": 2, "to": 5, "label": "1:N" }
        ]
    }
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate architecture data from code annotations')
    parser.add_argument('--api', default='/home/melchior/speech3/api', help='Path to API source')
    parser.add_argument('--app', default='/home/melchior/speechscore_app/lib', help='Path to Flutter app source')
    parser.add_argument('--output', default='/home/melchior/speech3/frontend/static/architecture-data.json', help='Output JSON file')
    parser.add_argument('--no-defaults', action='store_true', help='Skip adding default components')
    args = parser.parse_args()
    
    arch = ArchitectureParser()
    
    print(f"Parsing API source: {args.api}")
    arch.parse_directory(args.api, ['.py'])
    
    print(f"Parsing App source: {args.app}")
    arch.parse_directory(args.app, ['.dart'])
    
    # Add defaults for unannotated components
    if not args.no_defaults:
        print("Adding default components for unannotated items...")
        add_default_components(arch)
    
    print(f"Total: {len(arch.components)} components, {len(arch.flows)} flows")
    
    output = arch.to_json()
    with open(args.output, 'w') as f:
        f.write(output)
    print(f"Written to {args.output}")

if __name__ == '__main__':
    main()
