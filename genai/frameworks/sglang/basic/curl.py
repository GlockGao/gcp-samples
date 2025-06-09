import subprocess, json
from sglang.utils import print_highlight


host = "localhost"
host = "10.128.0.28"
port = 30000

curl_command = f"""
curl -s http://{host}:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
"""

response = json.loads(subprocess.check_output(curl_command, shell=True))
print_highlight(response)