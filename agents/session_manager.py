# session_manager.py — 多会话管理（JSON 持久化 + LRU 排序）
import json, time, uuid
from pathlib import Path
from dataclasses import asdict, dataclass

_SESSIONS_FILE = Path(__file__).parent.parent / "sessions.json"


@dataclass
class Session:
    thread_id: str  # UUID[:8]，关联 LangGraph thread_id
    name: str
    created_at: float
    updated_at: float


class SessionManager:
    """sessions.json 持久化 + 按最后活跃时间 LRU 排序"""

    def __init__(self, f: Path = _SESSIONS_FILE):
        self._f = f
        self._data = self._load()

    def _load(self) -> dict:
        if self._f.exists():
            try: return json.loads(self._f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError): pass
        return {"sessions": [], "active_thread_id": "default"}

    def _save(self):
        self._f.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def active_id(self) -> str:
        return self._data.get("active_thread_id", "default")

    @property
    def active_session(self) -> Session | None:
        for s in self.list_sessions():
            if s.thread_id == self.active_id:
                return s
        return None

    def list_sessions(self) -> list[Session]:
        ss = [Session(**d) for d in self._data.get("sessions", []) if isinstance(d, dict)]
        ss.sort(key=lambda s: s.updated_at, reverse=True)
        return ss

    def create(self, name: str = None) -> Session:
        s = Session(uuid.uuid4().hex[:8],
                    name or f"session_{len(self.list_sessions()) + 1}",
                    time.time(), time.time())
        self._data.setdefault("sessions", []).append(asdict(s))
        self._data["active_thread_id"] = s.thread_id
        self._save()
        return s

    def switch(self, index: int) -> Session:
        ss = self.list_sessions()
        if not 1 <= index <= len(ss):
            raise ValueError(f"序号超出范围 (1-{len(ss)})")
        self._data["active_thread_id"] = ss[index - 1].thread_id
        self._save()
        return ss[index - 1]

    def switch_by_name(self, name: str) -> Session:
        ms = [s for s in self.list_sessions() if name.lower() in s.name.lower()]
        if not ms: raise ValueError(f"未找到: {name}")
        if len(ms) > 1: raise ValueError(f"多个匹配 '{name}': {[s.name for s in ms]}")
        self._data["active_thread_id"] = ms[0].thread_id
        self._save()
        return ms[0]

    def rename(self, new_name: str):
        for s in self._data.get("sessions", []):
            if s.get("thread_id") == self.active_id:
                s["name"] = new_name
                break
        self._save()

    def delete(self, index: int) -> str:
        ss = self.list_sessions()
        if not 1 <= index <= len(ss): raise ValueError(f"序号超出范围 (1-{len(ss)})")
        removed = ss[index - 1]
        dl = self._data.get("sessions", [])
        dl.pop(index - 1)
        self._data["sessions"] = dl
        if self._data.get("active_thread_id") == removed.thread_id:
            self._data["active_thread_id"] = dl[0]["thread_id"] if dl else "default"
        self._save()
        return removed.thread_id

    def touch(self, tid: str = None):
        t = tid or self.active_id
        for s in self._data.get("sessions", []):
            if s.get("thread_id") == t:
                s["updated_at"] = time.time()
                break
        self._save()
