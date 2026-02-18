#!/usr/bin/env python3
"""
Architecture Test Runner
Runs test scenarios, captures logs, generates sequence diagrams, creates HTML report
"""

import os
import sys
import json
import time
import uuid
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from logs_to_puml import LogParser, PlantUMLGenerator, RequestTrace

@dataclass
class TestStep:
    name: str
    description: str
    method: str = "GET"
    endpoint: str = ""
    data: Dict = field(default_factory=dict)
    files: Dict = field(default_factory=dict)
    expected_status: int = 200
    
@dataclass
class TestScenario:
    name: str
    description: str
    steps: List[TestStep] = field(default_factory=list)
    
@dataclass
class TestResult:
    scenario: str
    step: str
    success: bool
    status_code: int
    duration_ms: float
    request_id: str = ""
    response_summary: str = ""
    error: str = ""
    trace_file: str = ""

class ArchitectureTestRunner:
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = ""):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key or os.environ.get('SPEAKFIT_API_KEY', 'sk-5b7fd66025ead6b8731ef73b2c970f26')
        self.results: List[TestResult] = []
        self.log_buffer = []
        self.output_dir = Path('/home/melchior/speech3/frontend/static/test-reports')
        
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make API request with tracing headers"""
        url = f"{self.api_url}{endpoint}"
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {self.api_key}'
        headers['X-Request-ID'] = str(uuid.uuid4())[:8]
        
        start = time.time()
        response = requests.request(method, url, headers=headers, **kwargs)
        duration = (time.time() - start) * 1000
        
        return response, headers['X-Request-ID'], duration
        
    def run_scenario(self, scenario: TestScenario) -> List[TestResult]:
        """Run a test scenario and collect results"""
        print(f"\n▶ Running scenario: {scenario.name}")
        print(f"  {scenario.description}")
        
        results = []
        for step in scenario.steps:
            print(f"  ├─ {step.name}...", end=' ')
            
            try:
                kwargs = {}
                if step.data:
                    kwargs['json'] = step.data
                if step.files:
                    kwargs['files'] = step.files
                    
                response, req_id, duration = self._request(
                    step.method, 
                    step.endpoint,
                    **kwargs
                )
                
                success = response.status_code == step.expected_status
                
                # Summarize response
                try:
                    resp_data = response.json()
                    if isinstance(resp_data, dict):
                        summary = ', '.join(f'{k}={str(v)[:30]}' for k, v in list(resp_data.items())[:3])
                    else:
                        summary = str(resp_data)[:100]
                except:
                    summary = response.text[:100]
                    
                result = TestResult(
                    scenario=scenario.name,
                    step=step.name,
                    success=success,
                    status_code=response.status_code,
                    duration_ms=duration,
                    request_id=req_id,
                    response_summary=summary
                )
                
                print(f"{'✓' if success else '✗'} {response.status_code} ({duration:.0f}ms)")
                
            except Exception as e:
                result = TestResult(
                    scenario=scenario.name,
                    step=step.name,
                    success=False,
                    status_code=0,
                    duration_ms=0,
                    error=str(e)
                )
                print(f"✗ Error: {e}")
                
            results.append(result)
            self.results.append(result)
            
            # Small delay between requests
            time.sleep(0.2)
            
        return results
        
    def collect_logs(self, log_file: str = '/home/melchior/speech3/api/logs/api.log', 
                     last_lines: int = 500) -> str:
        """Collect recent log entries"""
        try:
            result = subprocess.run(
                ['tail', '-n', str(last_lines), log_file],
                capture_output=True, text=True
            )
            return result.stdout
        except:
            return ""
            
    def generate_report(self, scenarios: List[TestScenario]) -> str:
        """Generate HTML test report with embedded diagrams"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_dir = self.output_dir / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect and parse logs
        log_content = self.collect_logs()
        log_parser = LogParser()
        log_parser.parse_log_string(log_content)
        
        # Generate PlantUML diagrams
        generator = PlantUMLGenerator()
        traces_dir = report_dir / 'traces'
        traces_dir.mkdir(exist_ok=True)
        
        # Map request IDs to trace files
        trace_files = {}
        for req_id in set(r.request_id for r in self.results if r.request_id):
            if req_id in log_parser.traces:
                trace = log_parser.traces[req_id]
                puml = generator.generate_sequence_diagram(trace)
                filename = f'trace_{req_id}.puml'
                with open(traces_dir / filename, 'w') as f:
                    f.write(puml)
                trace_files[req_id] = filename
                
        # Generate HTML
        html = self._generate_html(scenarios, trace_files, timestamp)
        
        report_file = report_dir / 'index.html'
        with open(report_file, 'w') as f:
            f.write(html)
            
        # Update index.html with link to latest report
        index_html = self.output_dir / 'index.html'
        index_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>SpeakFit Test Reports</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #fff; margin: 0; padding: 40px; }}
        h1 {{ color: #4CAF50; }}
        .report-link {{ display: block; padding: 15px 20px; margin: 10px 0; background: #16213e; border-radius: 8px; color: #2196F3; text-decoration: none; border-left: 4px solid #4CAF50; }}
        .report-link:hover {{ background: #1a2540; }}
    </style>
</head>
<body>
    <h1>🧪 SpeakFit Test Reports</h1>
    <p>Architecture test reports with sequence diagrams</p>
    <div class="reports">
        <a href="{timestamp}/" class="report-link">Latest: {timestamp}</a>
    </div>
</body>
</html>'''
        with open(index_html, 'w') as f:
            f.write(index_content)
        
        # Write results JSON
        with open(report_dir / 'results.json', 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
            
        return str(report_file)
        
    def _generate_html(self, scenarios: List[TestScenario], trace_files: Dict[str, str], timestamp: str) -> str:
        """Generate HTML report content"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Build scenario sections
        scenario_html = []
        for scenario in scenarios:
            scenario_results = [r for r in self.results if r.scenario == scenario.name]
            scenario_passed = all(r.success for r in scenario_results)
            
            steps_html = []
            for result in scenario_results:
                status_class = 'pass' if result.success else 'fail'
                trace_link = ''
                if result.request_id and result.request_id in trace_files:
                    trace_link = f'<a href="traces/{trace_files[result.request_id]}" class="trace-link">📊 Trace</a>'
                    
                steps_html.append(f'''
                <tr class="{status_class}">
                    <td>{result.step}</td>
                    <td><span class="status-badge {status_class}">{'✓' if result.success else '✗'}</span></td>
                    <td>{result.status_code}</td>
                    <td>{result.duration_ms:.0f}ms</td>
                    <td class="response">{result.response_summary[:50] or result.error[:50]}</td>
                    <td>{trace_link}</td>
                </tr>
                ''')
                
            scenario_html.append(f'''
            <div class="scenario {'pass' if scenario_passed else 'fail'}">
                <h3>
                    <span class="status-badge {'pass' if scenario_passed else 'fail'}">{'✓' if scenario_passed else '✗'}</span>
                    {scenario.name}
                </h3>
                <p class="description">{scenario.description}</p>
                <table>
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Status</th>
                            <th>Code</th>
                            <th>Duration</th>
                            <th>Response</th>
                            <th>Trace</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(steps_html)}
                    </tbody>
                </table>
            </div>
            ''')
            
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>SpeakFit Architecture Test Report - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #fff; 
            margin: 0;
            padding: 20px;
        }}
        h1 {{ color: #4CAF50; margin-bottom: 5px; }}
        h2 {{ color: #888; margin-top: 30px; }}
        h3 {{ margin: 0 0 10px 0; }}
        .timestamp {{ color: #666; font-size: 14px; }}
        .summary {{ 
            display: flex; 
            gap: 20px; 
            margin: 20px 0;
            padding: 20px;
            background: #16213e;
            border-radius: 8px;
        }}
        .summary-item {{
            text-align: center;
            padding: 15px 30px;
            background: #1a1a2e;
            border-radius: 6px;
        }}
        .summary-item .value {{ font-size: 36px; font-weight: bold; }}
        .summary-item .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        .summary-item.passed .value {{ color: #4CAF50; }}
        .summary-item.failed .value {{ color: #e74c3c; }}
        .summary-item.rate .value {{ color: #2196F3; }}
        
        .scenario {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #888;
        }}
        .scenario.pass {{ border-color: #4CAF50; }}
        .scenario.fail {{ border-color: #e74c3c; }}
        .scenario .description {{ color: #888; margin: 10px 0; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #888; font-weight: normal; font-size: 12px; text-transform: uppercase; }}
        tr.pass {{ background: rgba(76, 175, 80, 0.1); }}
        tr.fail {{ background: rgba(231, 76, 60, 0.1); }}
        
        .status-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-badge.pass {{ background: #4CAF50; color: white; }}
        .status-badge.fail {{ background: #e74c3c; color: white; }}
        
        .response {{ font-family: monospace; font-size: 11px; color: #888; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }}
        .trace-link {{ color: #2196F3; text-decoration: none; }}
        .trace-link:hover {{ text-decoration: underline; }}
        
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #333; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🧪 SpeakFit Architecture Test Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <div class="summary">
        <div class="summary-item passed">
            <div class="value">{passed}</div>
            <div class="label">Passed</div>
        </div>
        <div class="summary-item failed">
            <div class="value">{failed}</div>
            <div class="label">Failed</div>
        </div>
        <div class="summary-item rate">
            <div class="value">{pass_rate:.0f}%</div>
            <div class="label">Pass Rate</div>
        </div>
        <div class="summary-item">
            <div class="value">{len(scenarios)}</div>
            <div class="label">Scenarios</div>
        </div>
    </div>
    
    <h2>Test Scenarios</h2>
    {''.join(scenario_html)}
    
    <div class="footer">
        <p>SpeakFit Architecture Test Runner | <a href="../" style="color:#2196F3">All Reports</a> | <a href="results.json" style="color:#2196F3">Raw JSON</a></p>
    </div>
</body>
</html>'''

# Define test scenarios
SCENARIOS = [
    TestScenario(
        name="Health Check",
        description="Verify API is running and responsive",
        steps=[
            TestStep("Check health endpoint", "Basic health check", "GET", "/v1/health"),
            TestStep("Check API version", "Get API info", "GET", "/"),
        ]
    ),
    TestScenario(
        name="Speech Analysis Flow",
        description="End-to-end speech analysis from upload to results",
        steps=[
            TestStep("List speeches", "Get current speech list", "GET", "/v1/speeches?limit=5"),
            TestStep("Get single speech", "Get first speech details", "GET", "/v1/speeches?limit=1"),
        ]
    ),
    TestScenario(
        name="Projects API",
        description="Test project management endpoints",
        steps=[
            TestStep("List projects", "Get all projects", "GET", "/v1/projects"),
        ]
    ),
    TestScenario(
        name="Pipeline Status",
        description="Check ML pipeline readiness",
        steps=[
            TestStep("Queue status", "Check job queue status", "GET", "/v1/queue"),
        ]
    ),
]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run architecture tests and generate report')
    parser.add_argument('--api', default='http://localhost:8000', help='API URL')
    parser.add_argument('--key', default='', help='API key')
    args = parser.parse_args()
    
    runner = ArchitectureTestRunner(api_url=args.api, api_key=args.key)
    
    print("=" * 60)
    print("SpeakFit Architecture Test Runner")
    print("=" * 60)
    
    for scenario in SCENARIOS:
        runner.run_scenario(scenario)
        
    print("\n" + "=" * 60)
    print("Generating report...")
    
    report_path = runner.generate_report(SCENARIOS)
    print(f"Report generated: {report_path}")
    
    # Summary
    total = len(runner.results)
    passed = sum(1 for r in runner.results if r.success)
    print(f"\nResults: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
