#!/usr/bin/env python3
"""
AIML Work Agent Web Dashboard Server
Runs an executive dashboard web application providing a full UI for the
Reinforcement Learning-powered AI/ML Work Agent.
"""

import os
import sys
import json
import time
import random
import ast
import operator

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Core Definitions ---

class AgentWorkEnv:
    TASK_TYPES = {
        0: "Knowledge / Web Research",
        1: "Machine Learning / Data Science",
        2: "Math / Logic Computation",
        3: "Task Management / Scheduling",
        4: "General Query / Direct Response"
    }

    ACTION_NAMES = {
        0: "API_SEARCH",
        1: "ML_TRAIN_PREDICT",
        2: "MATH_COMPUTE",
        3: "TASK_MANAGE",
        4: "SYNTHESIZE_RESPONSE"
    }


class RLAgent:
    def __init__(self, num_actions: int = 5, alpha: float = 0.15, gamma: float = 0.95):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.02
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def get_q_values(self, state: Tuple[int, int, int, int]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, int, int, int], greedy: bool = True) -> int:
        q_vals = self.get_q_values(state)
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    def train_step(self, state, action, reward, next_state, done=True):
        q_vals = self.get_q_values(state)
        target = reward if done else reward + self.gamma * np.max(self.get_q_values(next_state))
        q_vals[action] += self.alpha * (target - q_vals[action])

    def load_policy(self, filepath: str = "trained_agent_rl_policy.joblib"):
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", 0.02)
            print(f" Loaded RL Policy with {len(self.q_table)} states from '{filepath}'")

    def save_policy(self, filepath: str = "trained_agent_rl_policy.joblib"):
        joblib.dump({"q_table": self.q_table, "epsilon": self.epsilon}, filepath)


class WikipediaAPITool:
    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    @classmethod
    def query(cls, topic: str) -> Dict[str, Any]:
        cleaned = topic.strip().replace(" ", "_")
        url = f"{cls.BASE_URL}{requests.utils.quote(cleaned)}"
        headers = {"User-Agent": "AIMLWorkAgentDashboard/1.0 (academic-research@example.com)"}
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "success",
                    "title": data.get("title", topic),
                    "extract": data.get("extract", "No extract found."),
                    "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", url)
                }
            return {"status": "error", "message": f"Wikipedia returned status {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class MachineLearningTool:
    @staticmethod
    def train_classifier(dataset_name: str = "iris") -> Dict[str, Any]:
        if dataset_name.lower() == "wine":
            data = load_wine()
            name = "Wine Recognition"
        elif dataset_name.lower() == "cancer":
            data = load_breast_cancer()
            name = "Breast Cancer Diagnostic"
        else:
            data = load_iris()
            name = "Iris Species Classification"

        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        importances = dict(zip(X.columns, np.round(clf.feature_importances_, 4)))
        sorted_feat = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        model_filename = f"{dataset_name.lower()}_trained_model.joblib"
        joblib.dump(clf, model_filename)

        return {
            "dataset": name,
            "total_samples": len(X),
            "features": list(X.columns),
            "test_accuracy": round(acc, 4),
            "feature_importance_ranking": sorted_feat[:5],
            "saved_artifact": model_filename
        }


class MathComputeTool:
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod
    }

    @classmethod
    def evaluate(cls, expression: str) -> Dict[str, Any]:
        clean = expression.replace("^", "**").replace("x", "*").replace("$", "").replace(",", "")
        try:
            node = ast.parse(clean, mode='eval').body
            def _eval(n):
                if isinstance(n, ast.Constant):
                    return n.value
                elif isinstance(n, ast.BinOp):
                    op_type = type(n.op)
                    if op_type in cls.ALLOWED_OPS:
                        return cls.ALLOWED_OPS[op_type](_eval(n.left), _eval(n.right))
                elif isinstance(n, ast.UnaryOp):
                    op_type = type(n.op)
                    if op_type in cls.ALLOWED_OPS:
                        return cls.ALLOWED_OPS[op_type](_eval(n.operand))
                raise ValueError("Unsupported AST operation")
            return {"status": "success", "result": _eval(node)}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@dataclass
class TaskItem:
    id: str
    description: str
    priority: str = "Medium"
    status: str = "PENDING"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    completed_at: Optional[str] = None


class TaskManagerTool:
    def __init__(self, storage_path: str = "agent_tasks.json"):
        self.storage_path = storage_path
        self.tasks: Dict[str, TaskItem] = {}
        self.load()

    def add_task(self, description: str, priority: str = "Medium") -> TaskItem:
        task_id = f"TASK-{len(self.tasks) + 1:03d}"
        item = TaskItem(id=task_id, description=description, priority=priority)
        self.tasks[task_id] = item
        self.save()
        return item

    def update_status(self, task_id: str, status: str) -> bool:
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if status == "COMPLETED":
                self.tasks[task_id].completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save()
            return True
        return False

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self.tasks.values()]

    def save(self):
        with open(self.storage_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.tasks.items()}, f, indent=2)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.tasks = {k: TaskItem(**v) for k, v in data.items()}
            except Exception:
                self.tasks = {}


class AIMLWorkAgent:
    def __init__(self, rl_agent: RLAgent):
        self.rl = rl_agent
        self.wiki_tool = WikipediaAPITool()
        self.ml_tool = MachineLearningTool()
        self.math_tool = MathComputeTool()
        self.task_manager = TaskManagerTool()
        self.history = []

    def perceive_and_encode_state(self, prompt: str) -> Tuple[int, int, int, int]:
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["train", "model", "accuracy", "classifier", "predict", "dataset", "iris", "wine"]):
            task_type = 1
        elif any(w in p_lower for w in ["calculate", "math", "+", "-", "*", "/", "return", "compound", "sum", "compute"]):
            task_type = 2
        elif any(w in p_lower for w in ["task", "schedule", "todo", "assign", "backlog", "status", "deadline"]):
            task_type = 3
        elif any(w in p_lower for w in ["what is", "who is", "explain", "search", "lookup", "wiki", "history", "define"]):
            task_type = 0
        else:
            task_type = 4

        complexity = 3 if len(prompt.split()) > 18 else (2 if len(prompt.split()) > 8 else 1)
        urgency = 3 if any(w in p_lower for w in ["urgent", "asap", "critical"]) else 2
        return (task_type, complexity, urgency, 0)

    def execute_task(self, prompt: str, priority: str = "Medium") -> Dict[str, Any]:
        task_entry = self.task_manager.add_task(prompt, priority=priority)
        state = self.perceive_and_encode_state(prompt)
        q_vals = [float(v) for v in self.rl.get_q_values(state)]
        action = self.rl.select_action(state, greedy=True)
        action_name = AgentWorkEnv.ACTION_NAMES[action]

        tool_result = None
        actions_taken = [action_name]

        if action == 0:
            query = prompt.replace("what is", "").replace("search", "").replace("lookup", "").strip(" ?.")
            if not query:
                query = "Reinforcement learning"
            tool_result = self.wiki_tool.query(query)
        elif action == 1:
            ds = "wine" if "wine" in prompt.lower() else ("cancer" if "cancer" in prompt.lower() else "iris")
            tool_result = self.ml_tool.train_classifier(ds)
        elif action == 2:
            expr = prompt
            for kw in ["calculate", "compute", "what is", "evaluate"]:
                expr = expr.lower().replace(kw, "")
            tool_result = self.math_tool.evaluate(expr.strip(" ?."))
        elif action == 3:
            tool_result = {"tasks": self.task_manager.list_tasks()}

        # Mark completed
        self.task_manager.update_status(task_entry.id, "COMPLETED")
        actions_taken.append("SYNTHESIZE_RESPONSE")

        # Synthesize response
        response = self._synthesize(prompt, actions_taken, tool_result)

        record = {
            "task_id": task_entry.id,
            "prompt": prompt,
            "state": state,
            "q_values": q_vals,
            "actions": actions_taken,
            "result": tool_result,
            "response": response
        }
        self.history.append(record)
        return record

    def _synthesize(self, prompt: str, actions: List[str], tool_result: Any) -> str:
        lines = ["I have processed your instruction using the trained Reinforcement Learning policy:"]
        lines.append(f"• Workflow Executed: {' -> '.join(actions)}")
        if isinstance(tool_result, dict):
            if "dataset" in tool_result:
                lines.append(f"• ML Model Performance: Achieved {tool_result['test_accuracy'] * 100:.2f}% accuracy on {tool_result['dataset']}.")
                lines.append(f"• Saved Artifact: {tool_result['saved_artifact']}")
                lines.append(f"• Top Predictive Features: {tool_result['feature_importance_ranking'][:3]}")
            elif "extract" in tool_result:
                lines.append(f"• Summary ({tool_result['title']}):\n  '{tool_result['extract']}'")
                lines.append(f"• Source: {tool_result.get('source_url', 'Wikipedia')}")
            elif "result" in tool_result:
                lines.append(f"• Computation Result: {tool_result['result']}")
            elif "tasks" in tool_result:
                lines.append(f"• Current Task Backlog ({len(tool_result['tasks'])} items listed in Backlog tab).")
        return "\n".join(lines)


# --- Initialize Global Agent ---
rl_policy = RLAgent()
rl_policy.load_policy("trained_agent_rl_policy.joblib")
work_agent = AIMLWorkAgent(rl_agent=rl_policy)


# --- HTTP Handler ---
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class AgentRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence routine access logs
        pass

    def send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Templates not found")
        elif self.path == "/api/tasks":
            self.send_json(work_agent.task_manager.list_tasks())
        elif self.path == "/api/stats":
            self.send_json({
                "explored_states": len(work_agent.rl.q_table),
                "completed_tasks": sum(1 for t in work_agent.task_manager.tasks.values() if t.status == "COMPLETED")
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body) if body else {}

        if self.path == "/api/execute":
            prompt = data.get("prompt", "")
            priority = data.get("priority", "Medium")
            result = work_agent.execute_task(prompt, priority=priority)
            self.send_json(result)

        elif self.path == "/api/feedback":
            reward = float(data.get("reward", 5.0))
            task_id = data.get("task_id")
            # Update most recent action
            if work_agent.history:
                last = work_agent.history[-1]
                state = last["state"]
                action_name = last["actions"][0]
                action_idx = [k for k, v in AgentWorkEnv.ACTION_NAMES.items() if v == action_name][0]
                work_agent.rl.train_step(state, action_idx, reward, state, done=True)
                work_agent.rl.save_policy()
                new_q = float(work_agent.rl.get_q_values(state)[action_idx])
                self.send_json({"status": "success", "new_q": new_q})
            else:
                self.send_json({"status": "error", "message": "No task history to rate"}, 400)

        elif self.path == "/api/ml/train":
            dataset = data.get("dataset", "iris")
            result = work_agent.ml_tool.train_classifier(dataset)
            self.send_json(result)

        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8000):
    server = ThreadedHTTPServer(("127.0.0.1", port), AgentRequestHandler)
    print(f"[INFO] AIML Work Agent Dashboard running live at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
