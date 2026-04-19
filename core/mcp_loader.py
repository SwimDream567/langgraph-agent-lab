"""MCP 动态加载器 — 读取 .mcp.json，连接外部 MCP 服务器，返回 LangChain 工具

功能：
  - 读取项目根目录的 .mcp.json
  - 检测配置文件变化（mtime），支持热加载
  - 返回标准 LangChain BaseTool 列表（直接塞进 _build_tools）
  - 支持用户手动编辑 / AI 通过斜杠命令添加 MCP 服务器
  - 使用持久 session（工具调用复用同一连接，浏览器不会关闭）

使用：
  from core.mcp_loader import MCPManager, CONFIG_PATH

  mgr = MCPManager()
  await mgr.load()           # 连接所有 MCP 服务器
  tools = mgr.tools          # LangChain BaseTool 列表
  changed = mgr.check()      # 检查配置是否变化
  await mgr.reload()         # 热重载（先关旧连接再开新连接）

配置文件格式（.mcp.json）：
  {
    "mcpServers": {
      "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
        "transport": "stdio"
      },
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:\\\\code"],
        "transport": "stdio"
      }
    }
  }
"""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 配置文件路径（项目根目录）
CONFIG_PATH = Path(__file__).parent.parent / ".mcp.json"


class MCPManager:
    """MCP 服务器连接管理器（单例模式）

    职责：
      1. 读取 .mcp.json 配置
      2. 为每个 MCP 服务器创建**持久 session**
      3. 返回 LangChain BaseTool 列表（工具复用同一 session）
      4. 检测配置变化，支持热重载
      5. 管理连接生命周期

    关键设计：
      - langchain-mcp-adapters 0.1.0 的 MultiServerMCPClient.get_tools()
        会为每次工具调用创建新 session（浏览器会被反复开关）
      - 我们绕过它，直接用 create_session() 创建持久 session
      - 工具通过 session 参数绑定到持久连接，不会每次重建
    """

    def __init__(self):
        self._tools: list = []
        self._session_contexts: list = []  # 持久 session 的上下文管理器
        self._active = False      # 是否已连接
        self._last_mtime: float = 0
        self._last_config: dict = {}

    # ── 属性 ──────────────────────────────────────────────

    @property
    def tools(self) -> list:
        """当前已加载的 MCP 工具列表（LangChain BaseTool）"""
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        """当前已加载的 MCP 工具名列表"""
        return [t.name for t in self._tools]

    @property
    def is_active(self) -> bool:
        """是否有活跃的 MCP 连接"""
        return self._active and len(self._tools) > 0

    # ── 配置读取 ──────────────────────────────────────────

    def read_config(self) -> dict:
        """读取 .mcp.json，返回 mcpServers 字典"""
        if not CONFIG_PATH.exists():
            return {}
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("mcpServers", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[MCP] 配置文件读取失败: {e}")
            return {}

    def _get_mtime(self) -> float:
        """获取配置文件修改时间"""
        try:
            return os.path.getmtime(CONFIG_PATH) if CONFIG_PATH.exists() else 0
        except OSError:
            return 0

    def check_changed(self) -> bool:
        """检查配置文件是否发生了变化"""
        current = self._get_mtime()
        if current != self._last_mtime:
            # mtime 变了，再检查内容是否真的变了
            new_config = self.read_config()
            if new_config != self._last_config:
                return True
        return False

    # ── 连接管理 ──────────────────────────────────────────

    async def load(self) -> list:
        """连接所有 MCP 服务器，返回工具列表

        必须在 async 上下文中调用（与 Agent 主循环同一个 event loop）。
        创建持久 session — 工具调用复用同一连接，不会每次重建。
        """
        config = self.read_config()
        self._last_config = config
        self._last_mtime = self._get_mtime()

        # 动态注入 --output-dir（截图/日志保存到启动目录下的 playwright-output/）
        _cwd = os.environ.get("LAUNCH_DIR", str(Path.cwd()))
        _pw_dir = str(Path(_cwd) / "playwright-output")
        os.makedirs(_pw_dir, exist_ok=True)
        for _srv in config.values():
            if isinstance(_srv, dict) and "args" in _srv:
                _args = _srv["args"]
                if any("playwright" in str(a) for a in _args) and "--output-dir" not in _args:
                    _args.extend(["--output-dir", _pw_dir.replace("\\", "/")])

        if not config:
            self._tools = []
            return []

        try:
            from langchain_mcp_adapters.sessions import create_session
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ImportError:
            logger.warning("[MCP] langchain-mcp-adapters 未安装，跳过 MCP 加载")
            logger.warning("[MCP] 安装命令: pip install langchain-mcp-adapters mcp")
            return []

        # 关闭旧连接（如果有）
        await self._close()

        # 为每个服务器创建持久 session 并加载工具
        all_tools = []
        self._session_contexts = []

        for name, connection in config.items():
            try:
                # 创建持久 session（手动管理生命周期，不用 async with）
                ctx = create_session(connection)
                session = await ctx.__aenter__()
                await session.initialize()

                # 保持上下文管理器存活（不退出 async with）
                self._session_contexts.append(ctx)

                # 加载工具 — 传入 session（不是 None！）
                # 这样工具的 execute_tool 会走 else 分支直接用 session.call_tool()
                # 而不是每次 async with create_session() 创建新连接
                tools = await load_mcp_tools(
                    session=session,
                    connection=connection,
                    server_name=name,
                )
                all_tools.extend(tools)
                logger.info(f"[MCP] 已连接 {name}，加载 {len(tools)} 个工具（持久 session）")
            except Exception as e:
                logger.error(f"[MCP] 连接 {name} 失败: {e}")

        self._tools = all_tools
        self._tag_tool_descriptions(config)
        self._active = len(all_tools) > 0

        server_names = list(config.keys())
        logger.info(f"[MCP] 已连接 {len(server_names)} 个服务器，加载 {len(self._tools)} 个工具")
        return self._tools

    async def reload(self) -> list:
        """热重载：关闭旧连接，读取新配置，建立新连接"""
        logger.info("[MCP] 热重载中...")
        return await self.load()

    def _tag_tool_descriptions(self, config: dict):
        """给 MCP 工具的 description 加 [MCP:服务器名] 前缀，让 LLM 区分来源"""
        if not self._tools:
            return
        tag = f"[MCP:{','.join(config.keys())}]"
        for tool in self._tools:
            if hasattr(tool, 'description') and tool.description:
                if not tool.description.startswith("[MCP:"):
                    tool.description = f"{tag} {tool.description}"

    async def _close(self):
        """关闭当前所有 MCP 持久 session"""
        for ctx in self._session_contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session_contexts = []
        self._active = False
        self._tools = []

    # ── 配置编辑（供斜杠命令和 AI 工具调用） ──────────────

    def add_server(self, name: str, config: dict) -> str:
        """添加一个 MCP 服务器到配置文件（不立即连接，等待热加载）"""
        data = {}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}

        if "mcpServers" not in data:
            data["mcpServers"] = {}

        data["mcpServers"][name] = config

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return f"✅ MCP 服务器 '{name}' 已添加（下次对话自动生效，或输入 /mcp reload 立即生效）"

    def remove_server(self, name: str) -> str:
        """从配置文件删除一个 MCP 服务器"""
        if not CONFIG_PATH.exists():
            return "❌ 配置文件不存在"

        data = {}
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        servers = data.get("mcpServers", {})
        if name not in servers:
            return f"❌ 未找到 MCP 服务器: {name}（已有: {', '.join(servers.keys()) or '无'}）"

        del servers[name]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return f"✅ MCP 服务器 '{name}' 已删除（下次对话自动生效，或输入 /mcp reload 立即生效）"

    def get_status(self) -> str:
        """获取当前 MCP 状态（供斜杠命令显示）"""
        config = self.read_config()
        lines = [f"  {_A_('c')}MCP 服务器状态{_A_('0')}"]
        lines.append(f"  {'─' * 45}")

        if not config:
            lines.append("  (无配置)")
            return "\n".join(lines)

        for name, cfg in config.items():
            cmd = cfg.get("command", "?")
            args = " ".join(cfg.get("args", []))
            transport = cfg.get("transport", "stdio")
            active_mark = f"{_A_('g')}●{_A_('0')}" if self.is_active else f"{_A_('d')}○{_A_('0')}"
            lines.append(f"  {active_mark} {_A_('c')}{name}{_A_('0')}")
            lines.append(f"    {cmd} {args}")
            lines.append(f"    transport: {transport}")

        if self.is_active:
            lines.append(f"\n  {_A_('g')}已加载 {len(self._tools)} 个工具（持久 session）:{_A_('0')} {', '.join(self.tool_names[:10])}")
            if len(self.tool_names) > 10:
                lines.append(f"    ... 共 {len(self.tool_names)} 个")
        else:
            lines.append(f"\n  {_A_('y')}尚未连接（输入 /mcp reload 连接）{_A_('0')}")

        return "\n".join(lines)


# ── 模块级辅助 ──────────────────────────────────────────

def _A_(key: str) -> str:
    """延迟导入颜色常量（避免循环导入）"""
    from agents.ui.colors import A
    return A.get(key, "")


# 全局单例
mcp_manager = MCPManager()
