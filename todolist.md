Auto Research Agent — 详细 ToDo List

---
  Phase 1: 基础框架（核心必做）

---
  ✅ T1.1 创建 nanobot/research/__init__.py

  文件路径: nanobot/research/__init__.py

  做什么: 模块初始化文件，导出核心类

  代码内容:
  """Auto Research Agent - 自主研究模块"""

  from nanobot.research.runner import ResearchRunner
  from nanobot.research.types import ResearchPlan, ResearchResult, ResearchState

  __all__ = ["ResearchRunner", "ResearchPlan", "ResearchResult", "ResearchState"]

---
  ✅ T1.2 创建 nanobot/research/types.py

  文件路径: nanobot/research/types.py

  做什么: 定义研究相关的数据结构（dataclass）

  代码内容:
  from dataclasses import dataclass, field
  from datetime import datetime
  from enum import Enum
  from typing import Any

  class ResearchStatus(Enum):
      PLANNING = "planning"      # 规划中
      SEARCHING = "searching"    # 搜索中
      SYNTHESIZING = "synthesizing"  # 综合中
      ITERATING = "iterating"    # 迭代补充中
      COMPLETED = "completed"    # 已完成
      FAILED = "failed"          # 失败

  @dataclass
  class SubQuestion:
      """拆解后的子问题"""
      id: int
      question: str
      keywords: list[str]          # 搜索关键词组合
      priority: int = 1            # 优先级 1-5
      status: str = "pending"      # pending/searching/completed
      results: list[dict] = field(default_factory=list)  # 搜索结果

  @dataclass
  class ResearchPlan:
      """研究计划"""
      topic: str                          # 原始研究方向
      sub_questions: list[SubQuestion]    # 子问题列表
      created_at: datetime = field(default_factory=datetime.now)
      iteration: int = 0                  # 当前迭代轮次

  @dataclass
  class SearchResult:
      """单个搜索结果"""
      url: str
      title: str
      content: str              # 提取的正文
      source_type: str          # paper/news/blog/official
      credibility_score: float  # 可信度 0-1
      relevance_score: float    # 相关度 0-1
      recency_score: float      # 时效性 0-1
      final_score: float = 0.0  # 综合评分

  @dataclass
  class SynthesisResult:
      """信息综合结果"""
      findings: list[dict]            # 核心发现列表
      contradictions: list[dict]      # 矛盾点列表
      knowledge_gaps: list[str]       # 知识空白
      coverage_score: float = 0.0     # 信息覆盖度 0-1
      sources: list[SearchResult] = field(default_factory=list)

  @dataclass
  class ResearchResult:
      """最终研究结果"""
      topic: str
      status: ResearchStatus
      report: str | None = None       # Markdown 报告
      plan: ResearchPlan | None = None
      synthesis: SynthesisResult | None = None
      created_at: datetime = field(default_factory=datetime.now)
      completed_at: datetime | None = None
      iterations: int = 0
      total_sources: int = 0
      quality_score: float = 0.0      # 自评质量分

  @dataclass
  class ResearchConfig:
      """研究配置"""
      max_iterations: int = 3         # 最大迭代轮次
      max_sources_per_question: int = 10  # 每个子问题最多搜索结果数
      min_coverage_threshold: float = 0.7  # 最小覆盖度阈值
      search_timeout: int = 30        # 单次搜索超时（秒）
      depth: str = "normal"           # quick/normal/deep

  关键点:
  - 所有数据结构用 @dataclass 定义
  - ResearchStatus 用枚举确保状态一致性
  - 评分字段用 float 0-1 便于后续比较

---
  ✅ T1.3 创建 nanobot/research/planner.py

  文件路径: nanobot/research/planner.py

  做什么: 研究规划器，负责拆解研究方向为子问题

  核心方法:
  class ResearchPlanner:
      def __init__(self, provider: LLMProvider, model: str):
          self.provider = provider
          self.model = model

      async def plan(self, topic: str, depth: str = "normal") -> ResearchPlan:
          """
          输入: 用户研究方向（如 "LLM在医疗领域的应用"）
          输出: ResearchPlan（包含 3-5 个子问题 + 关键词组合）
    
          做什么:
          1. 构造 LLM prompt，让 LLM 拆解研究方向
          2. 使用 tool_call 让 LLM 返回结构化的子问题列表
          3. 解析 LLM 返回，构建 ResearchPlan 对象
          """
          pass

  LLM Prompt 设计:
  System: 你是一个研究规划专家。用户会给你一个研究方向，你需要：
  1. 将研究方向拆解为 3-5 个具体的子问题
  2. 为每个子问题生成 2-3 组搜索关键词（中英文）
  3. 按重要性排序

  User: 研究方向：{topic}
  研究深度：{depth}

  请调用 research_plan 工具返回规划结果。

  Tool Definition:
  RESEARCH_PLAN_TOOL = {
      "type": "function",
      "function": {
          "name": "research_plan",
          "description": "输出研究规划",
          "parameters": {
              "type": "object",
              "properties": {
                  "sub_questions": {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "question": {"type": "string"},
                              "keywords_zh": {"type": "array", "items": {"type": "string"}},
                              "keywords_en": {"type": "array", "items": {"type": "string"}},
                              "priority": {"type": "integer"},
                          }
                      }
                  }
              },
              "required": ["sub_questions"]
          }
      }
  }

  输出示例:
  ResearchPlan(
      topic="LLM在医疗领域的应用",
      sub_questions=[
          SubQuestion(
              id=1,
              question="LLM在医疗影像诊断中的应用现状",
              keywords=["LLM 医疗影像", "AI 医学影像诊断", "LLM medical imaging diagnosis"],
              priority=1
          ),
          SubQuestion(
              id=2,
              question="LLM辅助临床决策的案例研究",
              keywords=["LLM 临床决策支持", "AI 医生助手", "LLM clinical decision support"],
              priority=2
          ),
          # ...
      ]
  )

---
  ✅ T1.4 创建 nanobot/research/searcher.py

  文件路径: nanobot/research/searcher.py

  做什么: 搜索编排器，负责并行搜索、去重、评分

  核心方法:
  class SearchOrchestrator:
      def __init__(self, web_search_tool, web_fetch_tool, config: ResearchConfig):
          self.web_search = web_search_tool    # 复用现有工具
          self.web_fetch = web_fetch_tool      # 复用现有工具
          self.config = config

      async def search(self, plan: ResearchPlan) -> list[SearchResult]:
          """
          输入: ResearchPlan
          输出: list[SearchResult]（去重+评分后的结果）
    
          做什么:
          1. 并行搜索所有子问题（asyncio.gather）
          2. 对每个搜索结果调用 web_fetch 获取正文
          3. 去重（基于 URL）
          4. 评分（可信度 × 相关度 × 时效性）
          5. 返回 Top N 结果
          """
          pass
    
      async def _search_sub_question(self, sq: SubQuestion) -> list[SearchResult]:
          """搜索单个子问题"""
          # 1. 用所有关键词组合搜索
          # 2. 合并结果
          # 3. 调用 web_fetch 获取详情
          pass
    
      def _score_result(self, result: dict, sq: SubQuestion) -> SearchResult:
          """评分逻辑"""
          # 可信度: 根据域名判断（.gov/.edu 高，博客低）
          # 相关度: 标题/内容与关键词匹配度
          # 时效性: 发布时间（越新越高）
          pass
    
      def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
          """URL 去重，保留评分最高的"""
          pass

  评分规则:
  def _calculate_credibility(self, url: str) -> float:
      """根据域名判断可信度"""
      HIGH_CREDIBILITY = [".gov", ".edu", ".org", "arxiv.org", "nature.com", "sciencedirect.com"]
      MEDIUM_CREDIBILITY = ["medium.com", "substack.com", "知乎", "公众号"]

      for domain in HIGH_CREDIBILITY:
          if domain in url:
              return 0.9
      for domain in MEDIUM_CREDIBILITY:
          if domain in url:
              return 0.6
      return 0.5

  def _calculate_recency(self, content: str, publish_date: str | None) -> float:
      """时效性评分"""
      # 2024年内容: 1.0
      # 2023年: 0.8
      # 2022年: 0.6
      # 更早: 0.4
      pass

---
  ✅ T1.5 创建 nanobot/research/synthesizer.py

  文件路径: nanobot/research/synthesizer.py

  做什么: 信息综合器，负责聚合、矛盾检测、知识空白识别

  核心方法:
  class InformationSynthesizer:
      def __init__(self, provider: LLMProvider, model: str):
          self.provider = provider
          self.model = model

      async def synthesize(
          self,
          results: list[SearchResult],
          plan: ResearchPlan
      ) -> SynthesisResult:
          """
          输入: 搜索结果 + 研究计划
          输出: SynthesisResult（发现、矛盾、空白）
    
          做什么:
          1. 将所有搜索结果拼接成 prompt
          2. 让 LLM 提取核心发现
          3. 让 LLM 识别矛盾观点
          4. 让 LLM 判断知识空白
          5. 计算覆盖度分数
          """
          pass

  LLM Prompt:
  System: 你是一个信息分析专家。我会给你多篇文档，你需要：
  1. 提取核心发现（每个发现要有数据支撑和来源）
  2. 识别矛盾观点（哪些问题不同来源说法不一致）
  3. 判断知识空白（哪些问题信息不够充分）

  User:
  研究问题：{plan.topic}
  子问题列表：{plan.sub_questions}

  搜索结果：
  {formatted_results}

  请调用 synthesize 工具返回分析结果。

  Tool Definition:
  SYNTHESIZE_TOOL = {
      "type": "function",
      "function": {
          "name": "synthesize",
          "parameters": {
              "type": "object",
              "properties": {
                  "findings": {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "finding": {"type": "string"},
                              "evidence": {"type": "string"},
                              "source_urls": {"type": "array", "items": {"type": "string"}},
                          }
                      }
                  },
                  "contradictions": {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "topic": {"type": "string"},
                              "viewpoint_a": {"type": "string"},
                              "viewpoint_b": {"type": "string"},
                          }
                      }
                  },
                  "knowledge_gaps": {
                      "type": "array",
                      "items": {"type": "string"}
                  },
                  "coverage_assessment": {
                      "type": "object",
                      "properties": {
                          "sub_question_id": {"type": "integer"},
                          "coverage": {"type": "string", "enum": ["sufficient", "partial", "insufficient"]},
                      }
                  }
              }
          }
      }
  }

---
  ✅ T1.6 创建 nanobot/research/refiner.py

  文件路径: nanobot/research/refiner.py

  做什么: 迭代优化器，决定是否需要补充搜索

  核心方法:
  class ResearchRefiner:
      def __init__(self, provider: LLMProvider, model: str):
          self.provider = provider
          self.model = model

      async def refine(
          self,
          plan: ResearchPlan,
          synthesis: SynthesisResult
      ) -> ResearchPlan | None:
          """
          输入: 当前计划 + 综合结果
          输出: 新的 ResearchPlan（补充搜索方向）或 None（信息足够）
    
          做什么:
          1. 检查知识空白和覆盖度
          2. 如果覆盖度 < 阈值，生成补充搜索关键词
          3. 更新 ResearchPlan，添加新的子问题或关键词
          """
          pass
    
      def should_continue(self, synthesis: SynthesisResult, iteration: int, config: ResearchConfig) -> bool:
          """判断是否需要继续迭代"""
          if iteration >= config.max_iterations:
              return False
          if synthesis.coverage_score >= config.min_coverage_threshold:
              return False
          return True

---
  ✅ T1.7 创建 nanobot/research/reporter.py

  文件路径: nanobot/research/reporter.py

  做什么: 报告生成器，输出结构化 Markdown

  核心方法:
  class ReportGenerator:
      def __init__(self, provider: LLMProvider, model: str):
          self.provider = provider
          self.model = model

      async def generate(
          self,
          topic: str,
          synthesis: SynthesisResult,
          plan: ResearchPlan
      ) -> str:
          """
          输入: 综合结果 + 研究计划
          输出: Markdown 格式的研究报告
    
          做什么:
          1. 用 LLM 将 findings 组织成结构化报告
          2. 包含执行摘要、核心发现、矛盾点、知识空白、来源列表
          3. 自评质量分数
          """
          pass
    
      async def self_evaluate(self, report: str, synthesis: SynthesisResult) -> float:
          """让 LLM 给自己的报告打分"""
          pass

  报告模板:
  # {topic} 研究报告

  ## 执行摘要
  {一句话核心结论}

  ## 研究方法
  - 研究问题拆解: {n} 个子问题
  - 信息来源: {m} 篇文档
  - 迭代轮次: {iterations} 轮

  ## 核心发现

  ### 1. {finding_1_title}
  {finding_1_content}
  > 来源: [标题](url)

  ### 2. {finding_2_title}
  ...

  ## 矛盾与争议
  {contradictions}

  ## 知识空白
  {knowledge_gaps}

  ## 信息来源
  {source_list}

---
  *报告生成时间: {timestamp}*
  *质量自评: {quality_score}/10*

---
  ✅ T1.8 创建 nanobot/research/runner.py

  文件路径: nanobot/research/runner.py

  做什么: 研究流程编排器，串联所有组件

  核心方法:
  class ResearchRunner:
      def __init__(
          self,
          provider: LLMProvider,
          model: str,
          web_search_tool,
          web_fetch_tool,
          config: ResearchConfig | None = None,
      ):
          self.provider = provider
          self.model = model
          self.config = config or ResearchConfig()

          self.planner = ResearchPlanner(provider, model)
          self.searcher = SearchOrchestrator(web_search_tool, web_fetch_tool, self.config)
          self.synthesizer = InformationSynthesizer(provider, model)
          self.refiner = ResearchRefiner(provider, model)
          self.reporter = ReportGenerator(provider, model)
    
          self._active_research: dict[str, ResearchResult] = {}  # 研究ID -> 结果
    
      async def run(self, topic: str, depth: str = "normal") -> ResearchResult:
          """
          主流程:
          1. 规划
          2. 搜索
          3. 综合
          4. 迭代 (如果需要)
          5. 报告生成
    
          返回: ResearchResult
          """
          research_id = str(uuid.uuid4())[:8]
          result = ResearchResult(topic=topic, status=ResearchStatus.PLANNING)
          self._active_research[research_id] = result
    
          try:
              # Phase 1: 规划
              plan = await self.planner.plan(topic, depth)
              result.plan = plan
    
              # Phase 2-4: 搜索 + 迭代
              for iteration in range(self.config.max_iterations):
                  result.status = ResearchStatus.SEARCHING
                  search_results = await self.searcher.search(plan)
    
                  result.status = ResearchStatus.SYNTHESIZING
                  synthesis = await self.synthesizer.synthesize(search_results, plan)
    
                  # 判断是否需要继续
                  if not self.refiner.should_continue(synthesis, iteration, self.config):
                      break
    
                  result.status = ResearchStatus.ITERATING
                  plan = await self.refiner.refine(plan, synthesis)
                  if plan is None:
                      break
                  result.iterations = iteration + 1
    
              # Phase 5: 报告
              result.status = ResearchStatus.COMPLETED
              result.synthesis = synthesis
              result.report = await self.reporter.generate(topic, synthesis, plan)
              result.quality_score = await self.reporter.self_evaluate(result.report, synthesis)
              result.completed_at = datetime.now()
    
              return result
    
          except Exception as e:
              result.status = ResearchStatus.FAILED
              raise

---
  Phase 2: 工具集成

---
  ✅ T2.1 创建 nanobot/agent/tools/research.py

  文件路径: nanobot/agent/tools/research.py

  做什么: Research Tool 入口，供 Agent 调用

  核心代码:
  from nanobot.agent.tools.base import Tool
  from nanobot.research.runner import ResearchRunner
  from nanobot.research.types import ResearchConfig

  class ResearchTool(Tool):
      name = "research"
      description = (
          "启动自主研究任务。"
          "当用户需要深入了解某个话题、对比多个观点、生成研究报告时使用。"
          "返回研究状态和报告。"
      )
      parameters = {
          "type": "object",
          "properties": {
              "action": {
                  "type": "string",
                  "enum": ["start", "status", "list"],
                  "description": "start=启动研究, status=查看进度, list=列出历史"
              },
              "topic": {
                  "type": "string",
                  "description": "研究方向（仅 start 时需要）"
              },
              "depth": {
                  "type": "string",
                  "enum": ["quick", "normal", "deep"],
                  "description": "研究深度（默认 normal）"
              },
              "research_id": {
                  "type": "string",
                  "description": "研究ID（仅 status 时需要）"
              }
          },
          "required": ["action"]
      }

      def __init__(
          self,
          provider,
          model: str,
          web_search_tool,
          web_fetch_tool,
          config: ResearchConfig | None = None,
      ):
          self._runner = ResearchRunner(
              provider=provider,
              model=model,
              web_search_tool=web_search_tool,
              web_fetch_tool=web_fetch_tool,
              config=config,
          )
          self._results: dict[str, ResearchResult] = {}  # 缓存结果
    
      async def execute(
          self,
          action: str,
          topic: str | None = None,
          depth: str = "normal",
          research_id: str | None = None,
          **kwargs
      ) -> str:
          if action == "start":
              if not topic:
                  return "Error: topic is required for start action"
              result = await self._runner.run(topic, depth)
              self._results[result.id] = result
              return f"研究完成！\n\n{result.report}"
    
          elif action == "status":
              if not research_id:
                  return f"Error: research_id is required. Active: {list(self._results.keys())}"
              result = self._results.get(research_id)
              if not result:
                  return f"Error: research {research_id} not found"
              return f"状态: {result.status.value}\n迭代: {result.iterations}\n来源: {result.total_sources}"
    
          elif action == "list":
              if not self._results:
                  return "暂无研究记录"
              lines = [f"- {rid}: {r.topic} ({r.status.value})" for rid, r in self._results.items()]
              return "\n".join(lines)
    
          return f"Unknown action: {action}"

---
  ✅ T2.2 修改 nanobot/agent/loop.py

  文件路径: nanobot/agent/loop.py

  做什么: 在 _register_default_tools 中注册 research_tool

  修改位置: 在 _register_default_tools 方法末尾添加

  def _register_default_tools(self) -> None:
      # ... 现有代码 ...

      # 注册 research_tool
      from nanobot.research.types import ResearchConfig
      research_config = ResearchConfig()  # 可从 config 读取
      self.tools.register(ResearchTool(
          provider=self.provider,
          model=self.model,
          web_search_tool=self.tools.get("web_search"),
          web_fetch_tool=self.tools.get("web_fetch"),
          config=research_config,
      ))

---
  ✅ T2.3 修改 nanobot/config/schema.py

  文件路径: nanobot/config/schema.py

  做什么: 添加 ResearchConfig 到配置 schema

  添加内容:
  class ResearchConfig(Base):
      """研究模块配置"""
      enabled: bool = True
      max_iterations: int = 3
      max_sources_per_question: int = 10
      min_coverage_threshold: float = 0.7
      search_timeout: int = 30
      default_depth: str = "normal"

  class ToolsConfig(Base):
      # ... 现有字段 ...
      research: ResearchConfig = Field(default_factory=ResearchConfig)

---
  Phase 3: 斜杠命令支持

---
  ✅ T3.1 修改 nanobot/command/builtin.py

  文件路径: nanobot/command/builtin.py

  做什么: 添加 /research 命令

  添加内容:
  async def cmd_research(ctx: CommandContext) -> OutboundMessage | None:
      """启动或管理研究任务"""
      loop = ctx.loop
      args = ctx.args.strip()

      if not args:
          return OutboundMessage(
              channel=ctx.msg.channel,
              chat_id=ctx.msg.chat_id,
              content="用法: /research <研究方向>\n示例: /research LLM在医疗领域的应用"
          )
    
      # 获取 research_tool
      research_tool = loop.tools.get("research")
      if not research_tool:
          return OutboundMessage(
              channel=ctx.msg.channel,
              chat_id=ctx.msg.chat_id,
              content="研究模块未启用"
          )
    
      # 执行研究
      result = await research_tool.execute(action="start", topic=args)
      return OutboundMessage(
          channel=ctx.msg.channel,
          chat_id=ctx.msg.chat_id,
          content=result
      )

  # 在 register_builtin_commands 中注册
  def register_builtin_commands(router: CommandRouter) -> None:
      # ... 现有代码 ...
      router.exact("/research", cmd_research)

---
  Phase 4: 依赖更新

---
  ✅ T4.1 修改 pyproject.toml

  文件路径: pyproject.toml

  做什么: 确认依赖完整（现有依赖已足够）

  说明: 不需要新增依赖，现有 httpx、ddgs、anthropic 已覆盖需求

---
  Phase 5: 测试（可选）

---
  ✅ T5.1 创建 tests/research/test_planner.py

  做什么: 测试研究规划逻辑

  ✅ T5.2 创建 tests/research/test_searcher.py

  做什么: 测试搜索编排逻辑

  ✅ T5.3 创建 tests/research/test_runner.py

  做什么: 测试完整流程

---
  文件清单汇总

  ┌─────────────────────────────────┬──────────┬──────┐
  │              文件               │ 行数估算 │ 状态 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/__init__.py    │ ~10      │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/types.py       │ ~80      │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/planner.py     │ ~120     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/searcher.py    │ ~150     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/synthesizer.py │ ~150     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/refiner.py     │ ~100     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/reporter.py    │ ~150     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/research/runner.py      │ ~200     │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/agent/tools/research.py │ ~80      │ 新建 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/agent/loop.py           │ +10      │ 修改 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/config/schema.py        │ +15      │ 修改 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ nanobot/command/builtin.py      │ +30      │ 修改 │
  ├─────────────────────────────────┼──────────┼──────┤
  │ 总计                            │ ~1100 行 │      │
  └─────────────────────────────────┴──────────┴──────┘

---
  实现顺序建议

  T1.2 (types.py) → T1.3 (planner.py) → T1.4 (searcher.py) → T1.5 (synthesizer.py)
      → T1.6 (refiner.py) → T1.7 (reporter.py) → T1.8 (runner.py) → T1.1 (__init__.py)
      → T2.1 (research.py) → T2.2 (loop.py) → T2.3 (schema.py) → T3.1 (builtin.py)

---

 Auto Research Agent - 完整 ToDo List

  Phase 1：数据模型 & 配置

  ┌───────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │           文件            │                                         任务                                          │
  ├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │                           │ 定义 ResearchPlan（研究计划）/ SubQuestion（子问题）/ SearchResult（搜索结果）/       │
  │ nanobot/research/types.py │ SynthesisResult（综合结果）/ ReportMetrics（报告评估指标）/                           │
  │                           │ ResearchResult（最终结果）                                                            │
  ├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/config/schema.py  │ 添加 ResearchConfig：max_iterations / max_sources_per_question /                      │
  │                           │ enable_self_evaluation / evaluation_threshold                                         │
  └───────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 2：研究规划器（Planner）

  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │            文件             │                                        任务                                         │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/planner.py │ LLM 调用生成                                                                        │
  │                             │ ResearchPlan；将用户问题拆解为独立子问题；注入背景知识；识别交叉依赖关系            │
  └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 3：搜索编排器（Searcher）

  ┌──────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │             文件             │                                        任务                                        │
  ├──────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/searcher.py │ 并行执行多路搜索（Web/Tavily/自定义）；去重 + 相似度过滤；按质量评分；返回结构化   │
  │                              │ SearchResult                                                                       │
  └──────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 4：信息综合器（Synthesizer）

  ┌─────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
  │              文件               │                                    任务                                    │
  ├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/synthesizer.py │ 多源信息融合；矛盾检测与来源可信度分析；知识空白识别；生成 SynthesisResult │
  └─────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

---
  Phase 5：迭代优化器（Refiner）+ 自评估（Self-Evaluator）

  ┌──────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │             文件             │                                       任务                                        │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/refiner.py  │ 评估当前综合结果是否"足够"；决定是否需要补充搜索；识别下一轮搜索方向；管理迭代轮  │
  │                              │ 次                                                                                │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/evaluator.p │ Self-Evaluation：LLM 自评报告完整性/准确性/可读性；输出                           │
  │ y                            │ ReportMetrics；若评分低于阈值，自动触发重新研究或补充搜索                         │
  └──────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 6：报告生成器（Reporter）

  ┌──────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │             文件             │                                        任务                                        │
  ├──────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/reporter.py │ 将综合结果组织为结构化                                                             │
  │                              │ Markdown；支持多种输出格式；集成自评估结果，生成带质量评分的最终报告               │
  └──────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 7：流程编排器（Runner）

  ┌────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │            文件            │                                         任务                                         │
  ├────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ nanobot/research/runner.py │ 串联 Planner → Searcher → Synthesizer → Refiner → Evaluator →                        │
  │                            │ Reporter；管理迭代循环；异常处理与降级策略                                           │
  └────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

---
  Phase 8：Tool 入口 & 命令行

  ┌─────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │              文件               │                          任务                           │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ nanobot/agent/tools/research.py │ 注册为 nanobot Tool；暴露自然语言接口；流式输出研究进度 │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ nanobot/agent/loop.py           │ 注册 research_tool 到工具列表                           │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ nanobot/command/builtin.py      │ 添加 /research 斜杠命令；支持后台执行                   │
  └─────────────────────────────────┴─────────────────────────────────────────────────────────┘

---
  Phase 9：测试 & 文档

  ┌────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │                文件                │                                任务                                 │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ tests/test_research_*.py           │ 单元测试（Planner/Searcher/Synthesizer/Refiner/Reporter/Evaluator） │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ tests/test_research_integration.py │ 集成测试（完整流程 mock）                                           │
  ├────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ docs/research_agent.md             │ 使用文档：命令格式、参数配置、输出示例                              │
  └────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

---