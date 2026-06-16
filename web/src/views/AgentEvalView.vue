<template>
  <app-layout>
    <div class="eval-page">

      <!-- 页头 + 统计卡片 -->
      <div class="page-header">
        <h2>Agent 评测</h2>
        <div class="stats-row">
          <div class="stat-card">
            <div class="stat-num">{{ stats.total_snapshots ?? '—' }}</div>
            <div class="stat-label">总快照</div>
          </div>
          <div class="stat-card warn">
            <div class="stat-num">{{ stats.total_badcases ?? '—' }}</div>
            <div class="stat-label">Badcase</div>
          </div>
          <div class="stat-card danger">
            <div class="stat-num">{{ stats.pending_review ?? '—' }}</div>
            <div class="stat-label">待审查</div>
          </div>
        </div>
      </div>

      <!-- 标签页 -->
      <a-tabs v-model:activeKey="tab" class="main-tabs" @change="onTabChange">

        <!-- ── 全部快照 ── -->
        <a-tab-pane key="snapshots" tab="运行快照">
          <div class="filter-bar">
            <a-select
              v-model:value="snapFilter.run_status"
              placeholder="状态筛选"
              allow-clear
              style="width: 130px"
              @change="fetchSnaps(1)"
            >
              <a-select-option value="success">成功</a-select-option>
              <a-select-option value="failed">失败</a-select-option>
              <a-select-option value="max_iterations">超迭代</a-select-option>
            </a-select>
            <a-select
              v-model:value="snapFilter.is_badcase"
              placeholder="Badcase 筛选"
              allow-clear
              style="width: 140px"
              @change="fetchSnaps(1)"
            >
              <a-select-option :value="true">仅 Badcase</a-select-option>
              <a-select-option :value="false">非 Badcase</a-select-option>
            </a-select>
            <a-button @click="fetchSnaps(1)"><reload-outlined /> 刷新</a-button>
          </div>

          <a-spin :spinning="snapsLoading">
            <a-table
              :data-source="snaps"
              :pagination="{ current: snapPage, pageSize: 20, total: snapTotal, onChange: fetchSnaps }"
              row-key="id"
              size="small"
              :scroll="{ x: 900 }"
              :custom-row="(record) => ({ onClick: () => openDetail(record) })"
              class="snap-table"
            >
              <a-table-column title="时间" data-index="timestamp" width="160">
                <template #default="{ record }">
                  <span class="mono">{{ fmtTime(record.timestamp) }}</span>
                </template>
              </a-table-column>
              <a-table-column title="用户输入" data-index="user_input" ellipsis />
              <a-table-column title="状态" data-index="run_status" width="90">
                <template #default="{ record }">
                  <a-tag :color="statusColor(record.run_status)">{{ record.run_status }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="工具调用" data-index="tool_call_count" width="80" align="center" />
              <a-table-column title="Token 入/出" width="120" align="center">
                <template #default="{ record }">
                  <span class="token-hint">{{ record.total_input_tokens }} / {{ record.total_output_tokens }}</span>
                </template>
              </a-table-column>
              <a-table-column title="耗时" data-index="total_duration_ms" width="90" align="center">
                <template #default="{ record }">
                  {{ record.total_duration_ms ? (record.total_duration_ms / 1000).toFixed(1) + 's' : '—' }}
                </template>
              </a-table-column>
              <a-table-column title="首字" data-index="ttft_ms" width="80" align="center">
                <template #default="{ record }">
                  {{ record.ttft_ms ? record.ttft_ms.toFixed(0) + 'ms' : '—' }}
                </template>
              </a-table-column>
              <a-table-column title="Badcase" width="90" align="center">
                <template #default="{ record }">
                  <a-tag v-if="record.is_badcase" color="red">{{ record.badcase_category || 'badcase' }}</a-tag>
                  <span v-else class="muted">—</span>
                </template>
              </a-table-column>
            </a-table>
          </a-spin>
        </a-tab-pane>

        <!-- ── Badcase ── -->
        <a-tab-pane key="badcases" tab="Badcase 管理">
          <div class="filter-bar">
            <a-select
              v-model:value="bcFilter.badcase_status"
              placeholder="处理状态"
              allow-clear
              style="width: 130px"
              @change="fetchBadcases(1)"
            >
              <a-select-option value="pending">待审查</a-select-option>
              <a-select-option value="reviewed">已审查</a-select-option>
              <a-select-option value="fixed">已修复</a-select-option>
              <a-select-option value="regressed">已回归</a-select-option>
            </a-select>
            <a-select
              v-model:value="bcFilter.badcase_category"
              placeholder="失败类型"
              allow-clear
              style="width: 140px"
              @change="fetchBadcases(1)"
            >
              <a-select-option value="run_failure">运行失败</a-select-option>
              <a-select-option value="token_spike">Token 异常</a-select-option>
              <a-select-option value="excessive_retries">过多重试</a-select-option>
              <a-select-option value="low_score">评分不足</a-select-option>
            </a-select>
            <a-select
              v-model:value="bcFilter.semantic_category"
              placeholder="语义分类"
              allow-clear
              style="width: 150px"
              @change="fetchBadcases(1)"
            >
              <a-select-option value="retrieval_failure">检索失败</a-select-option>
              <a-select-option value="hallucination">幻觉</a-select-option>
              <a-select-option value="tool_error">工具错误</a-select-option>
              <a-select-option value="reasoning_error">推理错误</a-select-option>
              <a-select-option value="context_loss">上下文丢失</a-select-option>
              <a-select-option value="instruction_following">指令遵循</a-select-option>
              <a-select-option value="output_format">输出格式</a-select-option>
            </a-select>
            <a-select
              v-model:value="bcFilter.root_cause_auto"
              placeholder="根因分类"
              allow-clear
              style="width: 130px"
              @change="fetchBadcases(1)"
            >
              <a-select-option value="prompt">prompt</a-select-option>
              <a-select-option value="context">context</a-select-option>
              <a-select-option value="tool">tool</a-select-option>
              <a-select-option value="model">model</a-select-option>
              <a-select-option value="user_input">user_input</a-select-option>
            </a-select>
            <a-button @click="fetchBadcases(1)"><reload-outlined /> 刷新</a-button>
            <a-button :loading="classifyLoading" @click="runClassifyBatch">批量语义分类</a-button>
            <a-button v-if="bcFilter.root_cause_auto === 'context'" type="primary" ghost @click="openContextDiag">诊断召回</a-button>
            <a-button v-if="bcFilter.root_cause_auto === 'user_input'" type="primary" ghost @click="openUserInputAnalysis">用户输入分析</a-button>
            <a-button v-if="bcFilter.root_cause_auto === 'model'" type="primary" ghost @click="openModelAnalysis">模型分析</a-button>
          </div>

          <a-spin :spinning="bcLoading">
            <a-table
              :data-source="badcases"
              :pagination="{ current: bcPage, pageSize: 20, total: bcTotal, onChange: fetchBadcases }"
              row-key="id"
              size="small"
              :scroll="{ x: 1000 }"
              :custom-row="(record) => ({ onClick: () => openDetail(record) })"
              class="snap-table"
            >
              <a-table-column title="时间" data-index="timestamp" width="160">
                <template #default="{ record }">
                  <span class="mono">{{ fmtTime(record.timestamp) }}</span>
                </template>
              </a-table-column>
              <a-table-column title="用户输入" data-index="user_input" ellipsis />
              <a-table-column title="触发类型" data-index="badcase_category" width="110">
                <template #default="{ record }">
                  <a-tag color="red">{{ record.badcase_category }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="语义分类" data-index="semantic_category" width="120">
                <template #default="{ record }">
                  <a-tag v-if="record.semantic_category" :color="semCategoryColor(record.semantic_category)">
                    {{ record.semantic_category }}
                  </a-tag>
                  <span v-else class="muted">—</span>
                </template>
              </a-table-column>
              <a-table-column title="根因" width="130">
                <template #default="{ record }">
                  <template v-if="record.root_cause_auto">
                    <a-tag :color="rootCauseColor(record.root_cause_auto)">{{ record.root_cause_auto }}</a-tag>
                    <span v-if="record.root_cause_auto_confidence" :class="confidenceCls(record.root_cause_auto_confidence)" style="font-size:10px;margin-left:2px">{{ record.root_cause_auto_confidence }}</span>
                  </template>
                  <span v-else class="muted">—</span>
                </template>
              </a-table-column>
              <a-table-column title="处理状态" data-index="badcase_status" width="100">
                <template #default="{ record }">
                  <a-tag :color="bcStatusColor(record.badcase_status)">{{ record.badcase_status }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="工具调用" data-index="tool_call_count" width="80" align="center" />
              <a-table-column title="Token 入/出" width="120" align="center">
                <template #default="{ record }">
                  <span class="token-hint">{{ record.total_input_tokens }} / {{ record.total_output_tokens }}</span>
                </template>
              </a-table-column>
            </a-table>
          </a-spin>
        </a-tab-pane>

        <!-- ── 测试集管理 ── -->
        <a-tab-pane key="testcases" tab="测试集">
          <div class="filter-bar">
            <a-select
              v-model:value="tcFilter"
              placeholder="数据集类型"
              allow-clear
              style="width: 150px"
              @change="fetchTestCases"
            >
              <a-select-option value="golden">黄金集</a-select-option>
              <a-select-option value="regression">回归集</a-select-option>
              <a-select-option value="calibration">标定集</a-select-option>
            </a-select>
            <a-button type="primary" @click="tcModalOpen = true"><plus-outlined /> 新建</a-button>
          </div>

          <a-spin :spinning="tcLoading">
            <a-table :data-source="testCases" row-key="id" size="small" :pagination="false">
              <a-table-column title="类型" data-index="dataset_type" width="90">
                <template #default="{ record }">
                  <a-tag :color="tcTypeColor(record.dataset_type)">{{ record.dataset_type }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="名称" data-index="name" />
              <a-table-column title="用户输入" data-index="user_input" ellipsis />
              <a-table-column title="期望工具" width="160">
                <template #default="{ record }">
                  <span class="muted">{{ (record.expected_tools || []).join(', ') || '—' }}</span>
                </template>
              </a-table-column>
              <a-table-column title="Token 上限" data-index="token_budget" width="100" align="center">
                <template #default="{ record }">{{ record.token_budget ?? '—' }}</template>
              </a-table-column>
              <a-table-column title="操作" width="80" align="center">
                <template #default="{ record }">
                  <a-popconfirm title="确认删除？" ok-text="删除" cancel-text="取消" @confirm="deleteTc(record.id)">
                    <a-button type="text" size="small" danger><delete-outlined /></a-button>
                  </a-popconfirm>
                </template>
              </a-table-column>
            </a-table>
          </a-spin>
        </a-tab-pane>

        <!-- ── 评测任务 ── -->
        <a-tab-pane key="evalruns" tab="评测任务">
          <div class="filter-bar">
            <a-button type="primary" @click="runModalOpen = true"><play-circle-outlined /> 触发评测</a-button>
            <a-button @click="fetchEvalRunsList()"><reload-outlined /> 刷新</a-button>
          </div>

          <a-spin :spinning="erLoading">
            <a-table
              :data-source="evalRuns"
              :pagination="{ current: erPage, pageSize: 20, total: erTotal, onChange: fetchEvalRunsList }"
              row-key="id"
              size="small"
              :custom-row="(record) => ({ onClick: () => openEvalRun(record) })"
              class="snap-table"
            >
              <a-table-column title="名称" data-index="name" />
              <a-table-column title="数据集" data-index="dataset_type" width="100">
                <template #default="{ record }">
                  <a-tag v-if="record.dataset_type" :color="tcTypeColor(record.dataset_type)">{{ record.dataset_type }}</a-tag>
                  <span v-else class="muted">全部</span>
                </template>
              </a-table-column>
              <a-table-column title="状态" data-index="status" width="90">
                <template #default="{ record }">
                  <a-tag :color="erStatusColor(record.status)">{{ record.status }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="通过率" width="120" align="center">
                <template #default="{ record }">
                  <span v-if="record.pass_rate != null" :class="scoreClass(record.pass_rate)">
                    {{ (record.pass_rate * 100).toFixed(0) }}%
                  </span>
                  <span v-else class="muted">—</span>
                </template>
              </a-table-column>
              <a-table-column title="用例数" width="80" align="center">
                <template #default="{ record }">{{ record.total_cases }}</template>
              </a-table-column>
              <a-table-column title="通过/失败" width="100" align="center">
                <template #default="{ record }">
                  <span class="score-high">{{ record.passed_cases }}</span>
                  /
                  <span class="score-low">{{ record.failed_cases }}</span>
                </template>
              </a-table-column>
              <a-table-column title="创建时间" data-index="created_at" width="160">
                <template #default="{ record }">
                  <span class="mono">{{ fmtTime(record.created_at) }}</span>
                </template>
              </a-table-column>
              <a-table-column title="操作" width="80" align="center">
                <template #default="{ record }">
                  <a-popconfirm title="确认删除该评测任务及其所有快照？" ok-text="删除" cancel-text="取消" @confirm.stop="removeEvalRun(record)">
                    <a-button type="link" danger size="small" @click.stop>删除</a-button>
                  </a-popconfirm>
                </template>
              </a-table-column>
            </a-table>
          </a-spin>
        </a-tab-pane>

        <!-- ── 趋势 ── -->
        <a-tab-pane key="trends" tab="趋势">
          <a-spin :spinning="trendsLoading">
            <div v-if="trends.total_scored === 0" class="muted" style="padding: 32px 0; text-align:center">
              暂无评分数据，先触发一次评测任务以生成趋势数据
            </div>
            <template v-else>
              <!-- 当前均分卡片 -->
              <div class="trends-header">
                <div class="section-title" style="margin-top:0">当前各维度均分（最近 {{ trendsDays }} 天）</div>
                <a-select v-model:value="trendsDays" style="width:100px" @change="loadTrends" size="small">
                  <a-select-option :value="7">最近 7 天</a-select-option>
                  <a-select-option :value="30">最近 30 天</a-select-option>
                  <a-select-option :value="90">最近 90 天</a-select-option>
                </a-select>
              </div>
              <div class="dim-cards">
                <div v-for="(val, dim) in trends.overall" :key="dim" class="dim-card">
                  <div class="dim-name">{{ dim }}</div>
                  <a-progress
                    :percent="+(val * 100).toFixed(1)"
                    :stroke-color="progressColor(val)"
                    size="small"
                  />
                  <div class="dim-score" :class="scoreClass(val)">{{ (val * 100).toFixed(1) }}%</div>
                </div>
              </div>
              <div class="section-title">按日分数明细（共 {{ trends.total_scored }} 条已评分快照）</div>
              <a-table
                :data-source="trends.daily"
                row-key="date"
                size="small"
                :pagination="false"
              >
                <a-table-column title="日期" data-index="date" width="120" />
                <a-table-column
                  v-for="dim in trendDims"
                  :key="dim"
                  :title="dim"
                  :data-index="['scores', dim]"
                  align="center"
                  width="130"
                >
                  <template #default="{ record }">
                    <span
                      v-if="record.scores && record.scores[dim] != null"
                      :class="scoreClass(record.scores[dim])"
                    >{{ (record.scores[dim] * 100).toFixed(1) }}%</span>
                    <span v-else class="muted">—</span>
                  </template>
                </a-table-column>
              </a-table>
            </template>
          </a-spin>
        </a-tab-pane>

        <!-- ── 优化建议 ── -->
        <a-tab-pane key="optimize" tab="优化建议">
          <div class="filter-bar">
            <a-select
              v-model:value="optFilter"
              placeholder="全部状态"
              allow-clear
              style="width: 130px"
              @change="fetchOptimizations(1)"
            >
              <a-select-option value="pending">待审批</a-select-option>
              <a-select-option value="approved">已批准</a-select-option>
              <a-select-option value="rejected">已拒绝</a-select-option>
            </a-select>
            <a-button @click="fetchOptimizations(1)"><reload-outlined /> 刷新</a-button>
            <a-button type="primary" @click="optModalOpen = true"><plus-outlined /> 触发优化</a-button>
          </div>

          <a-spin :spinning="optLoading">
            <a-table
              :data-source="optimizations"
              :pagination="{ current: optPage, pageSize: 20, total: optTotal, onChange: fetchOptimizations }"
              row-key="id"
              size="small"
              class="snap-table"
            >
              <a-table-column title="分类" data-index="category" width="140">
                <template #default="{ record }">
                  <a-tag :color="semCategoryColor(record.category)">{{ record.category }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="状态" data-index="status" width="90">
                <template #default="{ record }">
                  <a-tag :color="proposalStatusColor(record.status)">{{ record.status }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="候选数" width="80" align="center">
                <template #default="{ record }">{{ (record.proposals || []).length }}</template>
              </a-table-column>
              <a-table-column title="创建时间" data-index="created_at" width="160">
                <template #default="{ record }">
                  <span class="mono">{{ fmtTime(record.created_at) }}</span>
                </template>
              </a-table-column>
              <a-table-column title="操作" width="180" align="center">
                <template #default="{ record }">
                  <a-space>
                    <a-button size="small" @click.stop="openOptDetail(record)">详情</a-button>
                    <a-popconfirm v-if="record.status === 'pending'" title="批准此方案？" ok-text="批准" cancel-text="取消" @confirm="approveProposal(record.id)">
                      <a-button size="small" type="primary">批准</a-button>
                    </a-popconfirm>
                    <a-popconfirm v-if="record.status === 'pending'" title="拒绝此方案？" ok-text="拒绝" cancel-text="取消" @confirm="rejectProposal(record.id)">
                      <a-button size="small" danger>拒绝</a-button>
                    </a-popconfirm>
                  </a-space>
                </template>
              </a-table-column>
            </a-table>
          </a-spin>
        </a-tab-pane>

        <!-- ── 工具健康度 ── -->
        <a-tab-pane key="toolhealth" tab="工具健康度">
          <div class="filter-bar">
            <a-select v-model:value="thDays" style="width:130px" @change="fetchToolHealth">
              <a-select-option :value="1">最近 1 天</a-select-option>
              <a-select-option :value="7">最近 7 天</a-select-option>
              <a-select-option :value="30">最近 30 天</a-select-option>
            </a-select>
            <a-button @click="fetchToolHealth"><reload-outlined /> 刷新</a-button>
          </div>
          <a-spin :spinning="thLoading">
            <a-table
              v-if="toolHealth.length"
              :data-source="toolHealth"
              row-key="tool_name"
              size="small"
              :pagination="false"
            >
              <a-table-column title="工具" data-index="tool_name" width="160" />
              <a-table-column title="调用次数" data-index="total_calls" width="100" align="center" />
              <a-table-column title="错误数" data-index="error_calls" width="100" align="center" />
              <a-table-column title="错误率" data-index="error_rate" width="100" align="center">
                <template #default="{ record }">
                  <span :class="errorRateCls(record.error_rate)">
                    {{ (record.error_rate * 100).toFixed(1) }}%
                  </span>
                </template>
              </a-table-column>
              <a-table-column title="严重度" width="90" align="center">
                <template #default="{ record }">
                  <a-tag :color="record.severity">{{ record.severity }}</a-tag>
                </template>
              </a-table-column>
              <a-table-column title="错误示例" min-width="240">
                <template #default="{ record }">
                  <div v-if="record.sample_errors && record.sample_errors.length">
                    <div v-for="(err, ei) in record.sample_errors" :key="ei" class="err-sample">{{ err }}</div>
                  </div>
                  <span v-else class="muted">—</span>
                </template>
              </a-table-column>
            </a-table>
            <div v-else-if="!thLoading" class="muted" style="padding:32px 0;text-align:center">
              暂无工具调用记录
            </div>
          </a-spin>
        </a-tab-pane>

      </a-tabs>
    </div>

    <!-- ============================================================
         快照详情 Drawer
    ============================================================ -->
    <a-modal
      v-model:open="drawerOpen"
      title="运行快照详情"
      width="780"
      :footer="null"
      :destroy-on-close="true"
      @cancel="detail = null"
    >
      <a-spin :spinning="detailLoading">
        <template v-if="detail">
          <!-- 基础信息 -->
          <a-descriptions :column="2" size="small" bordered class="detail-desc">
            <a-descriptions-item label="状态">
              <a-tag :color="statusColor(detail.run_status)">{{ detail.run_status }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="时间">{{ fmtTime(detail.timestamp) }}</a-descriptions-item>
            <a-descriptions-item label="Token 入/出">
              {{ detail.total_input_tokens }} / {{ detail.total_output_tokens }}
            </a-descriptions-item>
            <a-descriptions-item v-if="detail.ttft_ms != null" label="首字时延">
              {{ detail.ttft_ms.toFixed(0) + ' ms' }}
            </a-descriptions-item>
            <a-descriptions-item label="总耗时">
              {{ detail.total_duration_ms ? (detail.total_duration_ms / 1000).toFixed(2) + ' s' : '—' }}
            </a-descriptions-item>
            <a-descriptions-item label="重试次数">{{ detail.retry_count }}</a-descriptions-item>
          </a-descriptions>

          <!-- 用户输入 -->
          <div class="section-title">用户输入</div>
          <div class="input-box">{{ detail.user_input }}</div>

          <!-- 工具调用链 -->
          <div class="section-title">工具调用链 ({{ detail.tool_call_chain.length }})</div>
          <div v-if="!detail.tool_call_chain.length" class="muted" style="margin-bottom:12px">无工具调用</div>
          <a-collapse v-else accordion class="tool-chain">
            <a-collapse-panel
              v-for="tc in detail.tool_call_chain"
              :key="tc.order"
              :header="toolHeader(tc)"
            >
              <div class="tool-section-label">参数</div>
              <pre class="code-block">{{ JSON.stringify(tc.params, null, 2) }}</pre>
              <div class="tool-section-label" style="margin-top:8px">返回值</div>
              <pre class="code-block result">{{ tc.result }}</pre>
              <div v-if="tc.error" class="error-hint">⚠ 工具报错</div>
              <div class="tool-meta">耗时 {{ tc.duration_ms != null ? tc.duration_ms.toFixed(0) + ' ms' : '—' }}</div>
            </a-collapse-panel>
          </a-collapse>

          <!-- LLM 调用 -->
          <div class="section-title">LLM 调用 ({{ detail.llm_calls.length }})</div>
          <div class="llm-calls">
            <div v-for="(c, i) in detail.llm_calls" :key="i" class="llm-call-row">
              <span class="mono">#{{ i + 1 }}</span>
              <span class="muted">{{ c.model }}</span>
              <span>↑ {{ c.input_tokens }}</span>
              <span>↓ {{ c.output_tokens }}</span>
            </div>
          </div>

          <!-- 最终回答 -->
          <div class="section-title">最终回答</div>
          <div class="response-box">{{ detail.final_response || '（无内容）' }}</div>

          <!-- 语义分类 -->
          <template v-if="detail.semantic_category">
            <div class="section-title">语义分类</div>
            <a-tag :color="semCategoryColor(detail.semantic_category)" style="margin-bottom:12px">
              {{ detail.semantic_category }}
            </a-tag>
          </template>

          <!-- 根因分类 -->
          <template v-if="detail.root_cause_auto">
            <div class="section-title">根因分类（自动）</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
              <a-tag :color="rootCauseColor(detail.root_cause_auto)" style="font-size:13px">
                {{ detail.root_cause_auto }}
              </a-tag>
              <a-tag v-if="detail.root_cause_auto_confidence" :color="confidenceTagColor(detail.root_cause_auto_confidence)">
                {{ detail.root_cause_auto_confidence }}
              </a-tag>
              <span v-if="detail.root_cause_auto_reason" class="muted" style="font-size:12px">{{ detail.root_cause_auto_reason }}</span>
            </div>
          </template>

          <!-- Badcase 标注 -->
          <template v-if="detail.is_badcase">
            <a-divider />
            <div class="section-title" style="display:flex;justify-content:space-between;align-items:center">
              <span>Badcase 标注</span>
              <a-button size="small" @click="openPromoteModal(detail.id)">提升为测试用例</a-button>
            </div>
            <a-form layout="vertical" style="margin-top:8px">
              <a-form-item label="处理状态">
                <a-select v-model:value="annotation.badcase_status" style="width:160px">
                  <a-select-option value="pending">待审查</a-select-option>
                  <a-select-option value="reviewed">已审查</a-select-option>
                  <a-select-option value="fixed">已修复</a-select-option>
                  <a-select-option value="regressed">已回归</a-select-option>
                </a-select>
              </a-form-item>
              <a-form-item label="根因分类">
                <a-input v-model:value="annotation.root_cause" placeholder="如：prompt 歧义 / 工具超时 / 模型幻觉" />
              </a-form-item>
              <a-form-item label="修复备注">
                <a-textarea v-model:value="annotation.fix_notes" :rows="3" placeholder="记录修复思路或关联 issue" />
              </a-form-item>
              <a-button type="primary" :loading="annotating" @click="saveAnnotation">保存标注</a-button>
            </a-form>
          </template>
        </template>
      </a-spin>
    </a-modal>

    <!-- ============================================================
         新建测试用例 Modal
    ============================================================ -->
    <a-modal
      v-model:open="tcModalOpen"
      title="新建测试用例"
      ok-text="保存"
      cancel-text="取消"
      :confirm-loading="tcSaving"
      @ok="saveTc"
      width="520"
      :destroy-on-close="true"
    >
      <a-form layout="vertical" style="margin-top:12px">
        <a-form-item label="数据集类型" required>
          <a-select v-model:value="tcForm.dataset_type" style="width:100%">
            <a-select-option value="golden">黄金集</a-select-option>
            <a-select-option value="regression">回归集</a-select-option>
            <a-select-option value="calibration">标定集</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="用例名称" required>
          <a-input v-model:value="tcForm.name" placeholder="简短描述这条用例的场景" />
        </a-form-item>
        <a-form-item label="用户输入" required>
          <a-textarea v-model:value="tcForm.user_input" :rows="3" />
        </a-form-item>
        <a-form-item label="期望工具（逗号分隔）">
          <a-input v-model:value="tcForm.expected_tools_str" placeholder="web_search, read_file" />
        </a-form-item>
        <a-form-item label="期望关键词（逗号分隔）">
          <a-input v-model:value="tcForm.expected_keywords_str" placeholder="价格, 上市时间" />
        </a-form-item>
        <a-form-item label="Token 预算上限">
          <a-input-number v-model:value="tcForm.token_budget" :min="0" style="width:140px" />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- ============================================================
         评测任务详情 Drawer
    ============================================================ -->
    <a-drawer
      v-model:open="erDrawerOpen"
      title="评测批次详情"
      placement="right"
      width="720"
      :body-style="{ padding: '16px 20px' }"
    >
      <a-spin :spinning="erDetailLoading">
        <template v-if="erDetail">
          <a-descriptions :column="2" size="small" bordered style="margin-bottom:16px">
            <a-descriptions-item label="状态">
              <a-tag :color="erStatusColor(erDetail.status)">{{ erDetail.status }}</a-tag>
            </a-descriptions-item>
            <a-descriptions-item label="数据集">{{ erDetail.dataset_type || '全部' }}</a-descriptions-item>
            <a-descriptions-item label="总用例">{{ erDetail.total_cases }}</a-descriptions-item>
            <a-descriptions-item label="通过率">
              <span v-if="erDetail.pass_rate != null" :class="scoreClass(erDetail.pass_rate)">
                {{ (erDetail.pass_rate * 100).toFixed(1) }}%
                （{{ erDetail.passed_cases }} / {{ erDetail.total_cases }}）
              </span>
              <span v-else class="muted">—</span>
            </a-descriptions-item>
            <a-descriptions-item label="创建时间">{{ fmtTime(erDetail.created_at) }}</a-descriptions-item>
            <a-descriptions-item label="完成时间">{{ fmtTime(erDetail.completed_at) }}</a-descriptions-item>
          </a-descriptions>

          <!-- 执行错误 -->
          <template v-if="erDetail.error">
            <a-alert type="error" show-icon style="margin-bottom:16px">
              <template #message>执行错误</template>
              <template #description>
                <pre style="white-space:pre-wrap;font-size:11px;margin:0;max-height:200px;overflow:auto">{{ erDetail.error }}</pre>
              </template>
            </a-alert>
          </template>

          <template v-if="erDetail.summary_scores && Object.keys(erDetail.summary_scores).length">
            <div class="section-title">维度均分</div>
            <div class="dim-cards" style="margin-bottom:16px">
              <div v-for="(val, dim) in erDetail.summary_scores" :key="dim" class="dim-card">
                <div class="dim-name">{{ dim }}</div>
                <a-progress :percent="+(val * 100).toFixed(1)" :stroke-color="progressColor(val)" size="small" />
              </div>
            </div>
          </template>

          <!-- 回归分析 -->
          <template v-if="erDetail.has_regression || erDetail.regression_diffs">
            <a-divider />
            <div class="section-title" style="color:#ff4d4f">
              {{ erDetail.has_regression ? '⚠ 回归检测：发现退化' : '回归分析' }}
            </div>
            <a-table
              v-if="erDetail.regression_diffs"
              :data-source="Object.entries(erDetail.regression_diffs).map(([dim, v]) => ({ dim, ...v }))"
              row-key="dim"
              size="small"
              :pagination="false"
              style="margin-bottom:16px"
            >
              <a-table-column title="维度" data-index="dim" />
              <a-table-column title="基线" data-index="baseline" align="center" width="90">
                <template #default="{ record }">{{ (record.baseline * 100).toFixed(1) }}%</template>
              </a-table-column>
              <a-table-column title="当前" data-index="current" align="center" width="90">
                <template #default="{ record }">{{ (record.current * 100).toFixed(1) }}%</template>
              </a-table-column>
              <a-table-column title="Delta" data-index="delta" align="center" width="90">
                <template #default="{ record }">
                  <span :class="deltaCls(record.delta)">
                    {{ record.delta >= 0 ? '+' : '' }}{{ (record.delta * 100).toFixed(1) }}%
                  </span>
                </template>
              </a-table-column>
              <a-table-column title="退化" data-index="regressed" align="center" width="70">
                <template #default="{ record }">
                  <span v-if="record.regressed" class="score-low">是</span>
                  <span v-else class="score-high">否</span>
                </template>
              </a-table-column>
            </a-table>
          </template>

          <!-- 与另一批次对比 -->
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
            <span style="font-size:12px;color:#888">对比另一批次：</span>
            <a-select v-model:value="compareRunId" allow-clear placeholder="选择批次" style="width:260px" size="small">
              <a-select-option v-for="r in completedRuns.filter(r => r.id !== erDetail.id)" :key="r.id" :value="r.id">
                {{ r.name }}
              </a-select-option>
            </a-select>
            <a-button size="small" :loading="compareLoading" @click="openCompare(compareRunId)">对比</a-button>
          </div>

          <div class="section-title">用例快照列表 <span class="muted" style="font-weight:normal;font-size:11px">（点击行查看完整详情）</span></div>
          <a-table
            :data-source="erDetail.snapshots || []"
            row-key="id"
            size="small"
            :pagination="false"
            :custom-row="(record) => ({ onClick: () => openDetail(record) })"
            class="snap-table"
          >
            <a-table-column title="用户输入" data-index="user_input" :width="160" ellipsis />
            <a-table-column title="Agent 回复" :width="200">
              <template #default="{ record }">
                <span class="muted" style="font-size:11px">{{ (record.final_response || '—').slice(0, 80) }}{{ (record.final_response || '').length > 80 ? '…' : '' }}</span>
              </template>
            </a-table-column>
            <a-table-column title="状态" data-index="run_status" width="100">
              <template #default="{ record }">
                <a-tag :color="statusColor(record.run_status)">{{ record.run_status }}</a-tag>
              </template>
            </a-table-column>
            <a-table-column title="通过" width="60" align="center">
              <template #default="{ record }">
                <span v-if="record.passed === true" class="score-high">✓</span>
                <span v-else-if="record.passed === false" class="score-low">✗</span>
                <span v-else class="muted">—</span>
              </template>
            </a-table-column>
            <a-table-column title="各维度分数" :width="200">
              <template #default="{ record }">
                <template v-if="record.scores && Object.keys(record.scores).length">
                  <a-tag
                    v-for="(val, dim) in record.scores" :key="dim"
                    :color="val >= 0.6 ? 'green' : 'red'"
                    style="font-size:10px;margin:1px"
                  >{{ dim }}: {{ (val * 100).toFixed(0) }}%</a-tag>
                </template>
                <span v-else class="muted">—</span>
              </template>
            </a-table-column>
            <a-table-column title="Token" width="100" align="center">
              <template #default="{ record }">
                <span class="token-hint">{{ record.total_input_tokens }} / {{ record.total_output_tokens }}</span>
              </template>
            </a-table-column>
          </a-table>
        </template>
      </a-spin>
    </a-drawer>

    <!-- ============================================================
         触发评测 Modal
    ============================================================ -->
    <a-modal
      v-model:open="runModalOpen"
      title="触发批量评测"
      ok-text="开始评测"
      cancel-text="取消"
      :confirm-loading="runSubmitting"
      @ok="submitEvalRun"
      width="440"
      :destroy-on-close="true"
    >
      <a-form layout="vertical" style="margin-top:12px">
        <a-form-item label="评测名称" required>
          <a-input v-model:value="runForm.name" placeholder="如：v1.2 回归测试" />
        </a-form-item>
        <a-form-item label="数据集类型">
          <a-select v-model:value="runForm.dataset_type" allow-clear placeholder="全部测试用例">
            <a-select-option value="golden">黄金集</a-select-option>
            <a-select-option value="regression">回归集</a-select-option>
            <a-select-option value="calibration">标定集</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="LLM Judge">
          <a-switch v-model:checked="runForm.use_judge" />
          <span class="field-hint" style="margin-left:8px">开启后会额外调用 LLM 对每条回答评分（较慢）</span>
        </a-form-item>
        <a-form-item label="对比基线">
          <a-select v-model:value="runForm.baseline_run_id" allow-clear placeholder="不设置基线">
            <a-select-option v-for="r in completedRuns" :key="r.id" :value="r.id">
              {{ r.name }} ({{ fmtTime(r.created_at) }})
            </a-select-option>
          </a-select>
          <div class="field-hint">选择后，完成时将自动进行回归分析</div>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- ============================================================
         提升为测试用例 Modal
    ============================================================ -->
    <a-modal
      v-model:open="promoteModalOpen"
      title="提升为测试用例"
      ok-text="保存"
      cancel-text="取消"
      :confirm-loading="promoteSaving"
      @ok="savePromote"
      width="480"
      :destroy-on-close="true"
    >
      <a-form layout="vertical" style="margin-top:12px">
        <a-form-item label="数据集类型" required>
          <a-select v-model:value="promoteForm.dataset_type" style="width:100%">
            <a-select-option value="golden">黄金集</a-select-option>
            <a-select-option value="regression">回归集</a-select-option>
            <a-select-option value="calibration">标定集</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="用例名称" required>
          <a-input v-model:value="promoteForm.name" placeholder="简短描述这条用例的场景" />
        </a-form-item>
        <a-form-item label="期望工具（逗号分隔）">
          <a-input v-model:value="promoteForm.expected_tools_str" placeholder="web_search, read_file" />
        </a-form-item>
        <a-form-item label="期望关键词（逗号分隔）">
          <a-input v-model:value="promoteForm.expected_keywords_str" placeholder="价格, 上市时间" />
        </a-form-item>
        <a-form-item label="Token 预算上限">
          <a-input-number v-model:value="promoteForm.token_budget" :min="0" style="width:140px" />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- ============================================================
         批次对比 Drawer
    ============================================================ -->
    <a-drawer
      v-model:open="compareDrawerOpen"
      title="评测批次维度对比"
      placement="right"
      width="600"
      :body-style="{ padding: '16px 20px' }"
      @close="compareResult = null; compareRunId = null"
    >
      <a-spin :spinning="compareLoading">
        <template v-if="compareResult">
          <a-table
            :data-source="Object.entries(compareResult.dimension_diffs || {}).map(([dim, v]) => ({ dim, ...v }))"
            row-key="dim"
            size="small"
            :pagination="false"
          >
            <a-table-column title="维度" data-index="dim" />
            <a-table-column title="Run A" data-index="score_a" align="center" width="90">
              <template #default="{ record }">{{ (record.score_a * 100).toFixed(1) }}%</template>
            </a-table-column>
            <a-table-column title="Run B" data-index="score_b" align="center" width="90">
              <template #default="{ record }">{{ (record.score_b * 100).toFixed(1) }}%</template>
            </a-table-column>
            <a-table-column title="Delta" data-index="delta" align="center" width="90">
              <template #default="{ record }">
                <span :class="deltaCls(record.delta)">
                  {{ record.delta >= 0 ? '+' : '' }}{{ (record.delta * 100).toFixed(1) }}%
                </span>
              </template>
            </a-table-column>
          </a-table>
        </template>
        <div v-else-if="!compareLoading" class="muted" style="text-align:center;padding:32px">请在评测详情中选择对比批次</div>
      </a-spin>
    </a-drawer>

    <!-- ============================================================
         触发优化 Modal
    ============================================================ -->
    <a-modal
      v-model:open="optModalOpen"
      title="触发 Prompt 优化"
      ok-text="提交"
      cancel-text="取消"
      :confirm-loading="optSubmitting"
      @ok="submitOptimization"
      width="480"
      :destroy-on-close="true"
    >
      <a-form layout="vertical" style="margin-top:12px">
        <a-form-item label="Badcase 语义分类" required>
          <a-select v-model:value="optForm.category" style="width:100%">
            <a-select-option value="retrieval_failure">检索失败</a-select-option>
            <a-select-option value="hallucination">幻觉</a-select-option>
            <a-select-option value="tool_error">工具错误</a-select-option>
            <a-select-option value="reasoning_error">推理错误</a-select-option>
            <a-select-option value="context_loss">上下文丢失</a-select-option>
            <a-select-option value="instruction_following">指令遵循</a-select-option>
            <a-select-option value="output_format">输出格式</a-select-option>
          </a-select>
        </a-form-item>
        <div class="field-hint" style="margin-top:-8px;margin-bottom:12px">
          系统将自动选取该分类下的代表性 Badcase 和黄金集用例进行优化
        </div>
      </a-form>
    </a-modal>

    <!-- ============================================================
         优化方案详情 Drawer
    ============================================================ -->
    <a-drawer
      v-model:open="optDetailOpen"
      title="优化候选方案详情"
      placement="right"
      width="680"
      :body-style="{ padding: '16px 20px' }"
      @close="optDetail = null"
    >
      <template v-if="optDetail">
        <a-descriptions :column="2" size="small" bordered style="margin-bottom:16px">
          <a-descriptions-item label="分类">
            <a-tag :color="semCategoryColor(optDetail.category)">{{ optDetail.category }}</a-tag>
          </a-descriptions-item>
          <a-descriptions-item label="状态">
            <a-tag :color="proposalStatusColor(optDetail.status)">{{ optDetail.status }}</a-tag>
          </a-descriptions-item>
          <a-descriptions-item label="创建时间">{{ fmtTime(optDetail.created_at) }}</a-descriptions-item>
          <a-descriptions-item label="候选方案数">{{ (optDetail.proposals || []).length }}</a-descriptions-item>
        </a-descriptions>

        <div class="section-title">候选 Prompt 方案（按评分排序）</div>
        <a-collapse accordion>
          <a-collapse-panel
            v-for="(p, i) in optDetail.proposals || []"
            :key="i"
            :header="`#${p.rank || i + 1}  均分 ${p.mean_score != null ? (p.mean_score * 100).toFixed(1) + '%' : '—'}`"
          >
            <div class="section-title" style="margin-top:0">改进说明</div>
            <p style="font-size:13px;color:#555">{{ p.rationale }}</p>
            <div class="section-title">候选 Prompt</div>
            <pre class="code-block" style="max-height:200px">{{ p.prompt }}</pre>
            <template v-if="p.scores && Object.keys(p.scores).length">
              <div class="section-title" style="margin-top:10px">维度评分</div>
              <div style="display:flex;flex-wrap:wrap;gap:8px">
                <span v-for="(v, d) in p.scores" :key="d" class="token-hint">
                  {{ d }}: <b :class="scoreClass(v)">{{ (v * 100).toFixed(1) }}%</b>
                </span>
              </div>
            </template>
          </a-collapse-panel>
        </a-collapse>

        <div v-if="optDetail.status === 'pending'" style="margin-top:16px;display:flex;gap:8px">
          <a-popconfirm title="批准此优化方案？" ok-text="批准" cancel-text="取消" @confirm="approveProposal(optDetail.id); optDetailOpen = false">
            <a-button type="primary">批准方案</a-button>
          </a-popconfirm>
          <a-popconfirm title="拒绝此方案？" ok-text="拒绝" cancel-text="取消" @confirm="rejectProposal(optDetail.id); optDetailOpen = false">
            <a-button danger>拒绝方案</a-button>
          </a-popconfirm>
        </div>
      </template>
    </a-drawer>

    <!-- ============================================================
         Context 诊断召回结果 Modal
    ============================================================ -->
    <a-modal
      v-model:open="ctxDiagOpen"
      title="诊断召回结果"
      width="760"
      :footer="null"
      :destroy-on-close="true"
    >
      <a-spin :spinning="ctxDiagLoading">
        <template v-if="ctxDiagResult">
          <div class="section-title" style="margin-top:0">汇总</div>
          <div style="display:flex;gap:12px;margin-bottom:16px">
            <div class="stat-card"><div class="stat-num score-high">{{ ctxDiagResult.summary.kb_gap }}</div><div class="stat-label">KB 缺失</div></div>
            <div class="stat-card warn"><div class="stat-num score-mid">{{ ctxDiagResult.summary.transient }}</div><div class="stat-label">暂时性失败</div></div>
            <div class="stat-card"><div class="stat-num">{{ ctxDiagResult.summary.unknown }}</div><div class="stat-label">无法判断</div></div>
          </div>

          <div class="section-title">逐条诊断</div>
          <a-collapse accordion>
            <a-collapse-panel
              v-for="item in ctxDiagResult.items"
              :key="item.snapshot_id"
              :header="`${(item.user_input || '—').slice(0, 60)}  (${item.queries.length} 条)`"
            >
              <a-table
                :data-source="item.queries"
                :row-key="(_r, i) => i"
                size="small"
                :pagination="false"
              >
                <a-table-column title="工具" data-index="tool_name" width="120" />
                <a-table-column title="查询词" data-index="query" width="150" ellipsis />
                <a-table-column title="当前检索数" data-index="now_count" width="100" align="center">
                  <template #default="{ record }">
                    <span v-if="record.now_count != null">{{ record.now_count }}</span>
                    <span v-else class="muted">—</span>
                  </template>
                </a-table-column>
                <a-table-column title="匹配 KB" data-index="kb_name" width="120">
                  <template #default="{ record }">
                    <span v-if="record.kb_name">{{ record.kb_name }}</span>
                    <span v-else class="muted">—</span>
                  </template>
                </a-table-column>
                <a-table-column title="判定" width="100">
                  <template #default="{ record }">
                    <a-tag :color="verdictColor(record.verdict)">{{ record.verdict }}</a-tag>
                  </template>
                </a-table-column>
              </a-table>
            </a-collapse-panel>
          </a-collapse>
        </template>
      </a-spin>
    </a-modal>

    <!-- ============================================================
         User-Input 分析结果 Modal
    ============================================================ -->
    <a-modal
      v-model:open="uiAnalysisOpen"
      title="用户输入分析结果"
      width="680"
      :footer="null"
      :destroy-on-close="true"
    >
      <a-spin :spinning="uiAnalysisLoading">
        <template v-if="uiAnalysisResult">
          <div class="section-title" style="margin-top:0">分析概况</div>
          <a-descriptions :column="2" size="small" bordered style="margin-bottom:16px">
            <a-descriptions-item label="分析输入数">{{ uiAnalysisResult.analyzed_count }}</a-descriptions-item>
          </a-descriptions>

          <template v-if="uiAnalysisResult.patterns && uiAnalysisResult.patterns.length">
            <div class="section-title">共同模式</div>
            <ul style="padding-left:18px;margin-bottom:16px">
              <li v-for="(p, i) in uiAnalysisResult.patterns" :key="i" style="margin-bottom:4px">{{ p }}</li>
            </ul>
          </template>

          <template v-if="uiAnalysisResult.prompt_suggestions && uiAnalysisResult.prompt_suggestions.length">
            <div class="section-title">Prompt 改进建议</div>
            <a-collapse accordion>
              <a-collapse-panel
                v-for="(s, i) in uiAnalysisResult.prompt_suggestions"
                :key="i"
                :header="`建议 ${i + 1}：${s.issue}`"
              >
                <pre class="code-block">{{ s.suggestion }}</pre>
              </a-collapse-panel>
            </a-collapse>
          </template>

          <div v-if="uiAnalysisResult.error" class="parse-error-hint">
            解析失败：<pre class="code-block">{{ uiAnalysisResult.raw }}</pre>
          </div>
        </template>
      </a-spin>
    </a-modal>

    <!-- ============================================================
         Model 分析结果 Modal
    ============================================================ -->
    <a-modal
      v-model:open="modelAnalysisOpen"
      title="模型分析结果"
      width="680"
      :footer="null"
      :destroy-on-close="true"
    >
      <a-spin :spinning="modelAnalysisLoading">
        <template v-if="modelAnalysisResult">
          <div class="section-title" style="margin-top:0">分析概况</div>
          <a-descriptions :column="2" size="small" bordered style="margin-bottom:16px">
            <a-descriptions-item label="分析记录数">{{ modelAnalysisResult.analyzed_count }}</a-descriptions-item>
            <a-descriptions-item label="平均 Token">{{ modelAnalysisResult.token_stats?.mean }}</a-descriptions-item>
            <a-descriptions-item label="最大 Token">{{ modelAnalysisResult.token_stats?.max }}</a-descriptions-item>
          </a-descriptions>

          <template v-if="modelAnalysisResult.pattern_report">
            <div class="section-title">失败模式分析</div>
            <div class="input-box" style="margin-bottom:16px">{{ modelAnalysisResult.pattern_report }}</div>
          </template>

          <template v-if="modelAnalysisResult.recommendations && modelAnalysisResult.recommendations.length">
            <div class="section-title">建议</div>
            <a-collapse accordion>
              <a-collapse-panel
                v-for="(r, i) in modelAnalysisResult.recommendations"
                :key="i"
                :header="`#${i + 1}  [${r.type}]`"
              >
                <div class="input-box">{{ r.detail }}</div>
              </a-collapse-panel>
            </a-collapse>
          </template>

          <div v-if="modelAnalysisResult.error" class="parse-error-hint">
            解析失败：<pre class="code-block">{{ modelAnalysisResult.raw }}</pre>
          </div>
        </template>
      </a-spin>
    </a-modal>

  </app-layout>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  ReloadOutlined, PlusOutlined, DeleteOutlined, PlayCircleOutlined,
} from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import {
  getSnapshots, getSnapshot, getBadcases, annotateBadcase,
  getTestCases, createTestCase, deleteTestCase, getEvalStats,
  triggerEvalRun, getEvalRuns, getEvalRun, deleteEvalRun, getTrends,
  promoteBadcase, classifyBadcasesBatch, getClassifyBatchStatus,
  compareEvalRuns, getEvalRunRegression,
  triggerOptimization, getOptimizations, updateOptimizationStatus,
  contextDiagnosis, getToolErrorStats,
  userInputAnalysis, getModelAnalysis,
} from '@/apis/agentEval'

// ── State ──────────────────────────────────────────────────────────────────
const tab = ref('snapshots')
const stats = ref({})

// Snapshots tab
const snaps = ref([])
const snapsLoading = ref(false)
const snapPage = ref(1)
const snapTotal = ref(0)
const snapFilter = reactive({ run_status: null, is_badcase: null })

// Badcase tab
const badcases = ref([])
const bcLoading = ref(false)
const bcPage = ref(1)
const bcTotal = ref(0)
const bcFilter = reactive({ badcase_status: null, badcase_category: null, semantic_category: null, root_cause_auto: null })

// Test cases tab
const testCases = ref([])
const tcLoading = ref(false)
const tcFilter = ref(null)
const tcModalOpen = ref(false)
const tcSaving = ref(false)
const tcForm = reactive({
  dataset_type: 'golden', name: '', user_input: '',
  expected_tools_str: '', expected_keywords_str: '', token_budget: null,
})

// Eval runs tab
const evalRuns = ref([])
const erLoading = ref(false)
const erPage = ref(1)
const erTotal = ref(0)
const erDrawerOpen = ref(false)
const erDetailLoading = ref(false)
const erDetail = ref(null)
const runModalOpen = ref(false)
const runSubmitting = ref(false)
const runForm = reactive({ name: '', dataset_type: null, use_judge: false, baseline_run_id: null })

// Trends tab
const trends = ref({ daily: [], overall: {}, total_scored: 0 })
const trendsLoading = ref(false)
const trendsDays = ref(30)
const trendDims = computed(() => Object.keys(trends.value.overall || {}))

// Drawer
const drawerOpen = ref(false)
const detailLoading = ref(false)
const detail = ref(null)
const annotating = ref(false)
const annotation = reactive({ badcase_status: 'pending', root_cause: '', fix_notes: '' })

// Batch classify
const classifyLoading = ref(false)
const classifyInterval = ref(null)

// Promote badcase
const promoteModalOpen = ref(false)
const promoteSaving = ref(false)
const promoteForm = reactive({ dataset_type: 'golden', name: '', expected_tools_str: '', expected_keywords_str: '', token_budget: null })
const promoteTargetId = ref(null)

// Eval run comparison
const completedRuns = ref([])
const compareRunId = ref(null)
const compareDrawerOpen = ref(false)
const compareLoading = ref(false)
const compareResult = ref(null)

// Optimization proposals
const optimizations = ref([])
const optLoading = ref(false)
const optPage = ref(1)
const optTotal = ref(0)
const optFilter = ref(null)
const optModalOpen = ref(false)
const optSubmitting = ref(false)
const optForm = reactive({ category: '', snapshot_ids: [] })
const optDetail = ref(null)
const optDetailOpen = ref(false)

// Tool health tab
const toolHealth = ref([])
const thLoading = ref(false)
const thDays = ref(7)

// Context diagnosis
const ctxDiagOpen = ref(false)
const ctxDiagLoading = ref(false)
const ctxDiagResult = ref(null)

// User-input analysis
const uiAnalysisOpen = ref(false)
const uiAnalysisLoading = ref(false)
const uiAnalysisResult = ref(null)

// Model analysis
const modelAnalysisOpen = ref(false)
const modelAnalysisLoading = ref(false)
const modelAnalysisResult = ref(null)

// ── Loaders ────────────────────────────────────────────────────────────────
async function fetchStats() {
  try { stats.value = await getEvalStats() } catch (_) {}
}

async function fetchSnaps(page = 1) {
  snapPage.value = page
  snapsLoading.value = true
  try {
    const res = await getSnapshots({ ...snapFilter, page, page_size: 20 })
    snaps.value = res.items
    snapTotal.value = res.total
  } catch (e) {
    message.error('加载快照失败：' + (e.message || ''))
  } finally {
    snapsLoading.value = false
  }
}

async function fetchBadcases(page = 1) {
  bcPage.value = page
  bcLoading.value = true
  try {
    const res = await getBadcases({ ...bcFilter, page, page_size: 20 })
    badcases.value = res.items
    bcTotal.value = res.total
  } catch (e) {
    message.error('加载 Badcase 失败：' + (e.message || ''))
  } finally {
    bcLoading.value = false
  }
}

async function fetchTestCases() {
  tcLoading.value = true
  try {
    testCases.value = await getTestCases(tcFilter.value)
  } catch (e) {
    message.error('加载测试集失败：' + (e.message || ''))
  } finally {
    tcLoading.value = false
  }
}

// ── Detail drawer ──────────────────────────────────────────────────────────
async function openDetail(record) {
  drawerOpen.value = true
  detailLoading.value = true
  detail.value = null
  try {
    detail.value = await getSnapshot(record.id)
    annotation.badcase_status = detail.value.badcase_status || 'pending'
    annotation.root_cause = detail.value.root_cause || ''
    annotation.fix_notes = detail.value.fix_notes || ''
  } catch (e) {
    message.error('加载详情失败：' + (e.message || ''))
  } finally {
    detailLoading.value = false
  }
}

async function saveAnnotation() {
  if (!detail.value) return
  annotating.value = true
  try {
    await annotateBadcase(detail.value.id, {
      badcase_status: annotation.badcase_status,
      root_cause: annotation.root_cause || null,
      fix_notes: annotation.fix_notes || null,
    })
    message.success('标注已保存')
    // Refresh badge counts
    fetchStats()
    fetchBadcases(bcPage.value)
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  } finally {
    annotating.value = false
  }
}

// ── Test case CRUD ─────────────────────────────────────────────────────────
async function saveTc() {
  if (!tcForm.name.trim() || !tcForm.user_input.trim()) {
    message.warning('名称和用户输入不能为空')
    return
  }
  tcSaving.value = true
  try {
    await createTestCase({
      dataset_type: tcForm.dataset_type,
      name: tcForm.name,
      user_input: tcForm.user_input,
      expected_tools: tcForm.expected_tools_str
        ? tcForm.expected_tools_str.split(',').map(s => s.trim()).filter(Boolean)
        : null,
      expected_keywords: tcForm.expected_keywords_str
        ? tcForm.expected_keywords_str.split(',').map(s => s.trim()).filter(Boolean)
        : null,
      token_budget: tcForm.token_budget || null,
    })
    tcModalOpen.value = false
    Object.assign(tcForm, { name: '', user_input: '', expected_tools_str: '', expected_keywords_str: '', token_budget: null })
    message.success('用例已创建')
    fetchTestCases()
  } catch (e) {
    message.error('创建失败：' + (e.message || ''))
  } finally {
    tcSaving.value = false
  }
}

async function deleteTc(id) {
  try {
    await deleteTestCase(id)
    message.success('已删除')
    fetchTestCases()
  } catch (e) {
    message.error('删除失败：' + (e.message || ''))
  }
}

async function removeEvalRun(record) {
  try {
    await deleteEvalRun(record.id)
    message.success('已删除')
    fetchEvalRunsList()
  } catch (e) {
    message.error('删除失败：' + (e.message || ''))
  }
}

// ── Eval runs ─────────────────────────────────────────────────────────────
async function fetchEvalRunsList(page = 1) {
  erPage.value = page
  erLoading.value = true
  try {
    const res = await getEvalRuns({ page, page_size: 20 })
    evalRuns.value = res.items
    erTotal.value = res.total
  } catch (e) {
    message.error('加载评测任务失败：' + (e.message || ''))
  } finally {
    erLoading.value = false
  }
}

async function openEvalRun(record) {
  erDrawerOpen.value = true
  erDetailLoading.value = true
  erDetail.value = null
  try {
    erDetail.value = await getEvalRun(record.id)
  } catch (e) {
    message.error('加载详情失败：' + (e.message || ''))
  } finally {
    erDetailLoading.value = false
  }
}

async function submitEvalRun() {
  if (!runForm.name.trim()) { message.warning('请填写评测名称'); return }
  runSubmitting.value = true
  try {
    await triggerEvalRun({
      name: runForm.name,
      dataset_type: runForm.dataset_type || null,
      use_judge: runForm.use_judge,
      baseline_run_id: runForm.baseline_run_id || null,
    })
    runModalOpen.value = false
    Object.assign(runForm, { name: '', dataset_type: null, use_judge: false, baseline_run_id: null })
    message.success('评测任务已提交，后台运行中')
    fetchEvalRunsList()
  } catch (e) {
    message.error('提交失败：' + (e.message || ''))
  } finally {
    runSubmitting.value = false
  }
}

// ── Batch classify ─────────────────────────────────────────────────────────
async function runClassifyBatch() {
  classifyLoading.value = true
  try {
    const res = await classifyBadcasesBatch()
    message.info(`已加入队列：${res.queued} 条`)
    classifyInterval.value = setInterval(async () => {
      try {
        const status = await getClassifyBatchStatus()
        if (status.unclassified === 0) {
          clearInterval(classifyInterval.value)
          classifyInterval.value = null
          classifyLoading.value = false
          message.success('批量语义分类完成')
          fetchBadcases(bcPage.value)
        }
      } catch (_) {}
    }, 3000)
  } catch (e) {
    classifyLoading.value = false
    message.error('触发失败：' + (e.message || ''))
  }
}

// ── Promote badcase ─────────────────────────────────────────────────────────
function openPromoteModal(id) {
  promoteTargetId.value = id
  Object.assign(promoteForm, { dataset_type: 'golden', name: '', expected_tools_str: '', expected_keywords_str: '', token_budget: null })
  promoteModalOpen.value = true
}

async function savePromote() {
  if (!promoteForm.name.trim()) { message.warning('用例名称不能为空'); return }
  promoteSaving.value = true
  try {
    await promoteBadcase(promoteTargetId.value, {
      dataset_type: promoteForm.dataset_type,
      name: promoteForm.name,
      expected_tools: promoteForm.expected_tools_str
        ? promoteForm.expected_tools_str.split(',').map(s => s.trim()).filter(Boolean)
        : null,
      expected_keywords: promoteForm.expected_keywords_str
        ? promoteForm.expected_keywords_str.split(',').map(s => s.trim()).filter(Boolean)
        : null,
      token_budget: promoteForm.token_budget || null,
    })
    promoteModalOpen.value = false
    message.success('已提升为测试用例')
    fetchTestCases()
  } catch (e) {
    message.error('提升失败：' + (e.message || ''))
  } finally {
    promoteSaving.value = false
  }
}

// ── Eval run comparison ────────────────────────────────────────────────────
async function loadCompletedRuns() {
  try {
    const res = await getEvalRuns({ page: 1, page_size: 50 })
    completedRuns.value = (res.items || []).filter(r => r.status === 'completed')
  } catch (_) {}
}

async function openCompare(runId) {
  if (!runId) return
  compareLoading.value = true
  compareDrawerOpen.value = true
  compareResult.value = null
  try {
    compareResult.value = await compareEvalRuns(erDetail.value.id, runId)
  } catch (e) {
    message.error('对比失败：' + (e.message || ''))
  } finally {
    compareLoading.value = false
  }
}

// ── Optimization proposals ─────────────────────────────────────────────────
async function fetchOptimizations(page = 1) {
  optPage.value = page
  optLoading.value = true
  try {
    const res = await getOptimizations({ status: optFilter.value, page, page_size: 20 })
    optimizations.value = res.items
    optTotal.value = res.total
  } catch (e) {
    message.error('加载优化建议失败：' + (e.message || ''))
  } finally {
    optLoading.value = false
  }
}

async function submitOptimization() {
  if (!optForm.category) { message.warning('请选择 Badcase 语义分类'); return }
  optSubmitting.value = true
  try {
    await triggerOptimization({ category: optForm.category, snapshot_ids: optForm.snapshot_ids })
    optModalOpen.value = false
    message.success('优化任务已提交，后台生成候选方案')
    fetchOptimizations()
  } catch (e) {
    message.error('提交失败：' + (e.message || ''))
  } finally {
    optSubmitting.value = false
  }
}

async function approveProposal(id) {
  try {
    await updateOptimizationStatus(id, 'approved')
    message.success('已批准')
    fetchOptimizations(optPage.value)
  } catch (e) {
    message.error('操作失败：' + (e.message || ''))
  }
}

async function rejectProposal(id) {
  try {
    await updateOptimizationStatus(id, 'rejected')
    message.success('已拒绝')
    fetchOptimizations(optPage.value)
  } catch (e) {
    message.error('操作失败：' + (e.message || ''))
  }
}

function openOptDetail(record) {
  optDetail.value = record
  optDetailOpen.value = true
}

// ── Trends ─────────────────────────────────────────────────────────────────
function onTabChange(key) {
  if (key === 'trends') loadTrends()
  else if (key === 'optimize') fetchOptimizations(1)
  else if (key === 'toolhealth') fetchToolHealth()
}

async function loadTrends() {
  trendsLoading.value = true
  try {
    trends.value = await getTrends(trendsDays.value)
  } catch (e) {
    message.error('加载趋势失败：' + (e.message || ''))
  } finally {
    trendsLoading.value = false
  }
}

// ── Tool health ──────────────────────────────────────────────────────────────
async function fetchToolHealth() {
  thLoading.value = true
  try {
    const res = await getToolErrorStats(thDays.value)
    toolHealth.value = res.tools || []
  } catch (e) {
    message.error('加载工具健康度失败：' + (e.message || ''))
  } finally {
    thLoading.value = false
  }
}

// ── Context diagnosis ────────────────────────────────────────────────────────
async function openContextDiag() {
  ctxDiagResult.value = null
  ctxDiagOpen.value = true
  ctxDiagLoading.value = true
  try {
    ctxDiagResult.value = await contextDiagnosis({ snapshot_ids: null, top_k: 5 })
  } catch (e) {
    message.error('诊断失败：' + (e.message || ''))
    ctxDiagOpen.value = false
  } finally {
    ctxDiagLoading.value = false
  }
}

// ── User-input analysis ──────────────────────────────────────────────────────
async function openUserInputAnalysis() {
  uiAnalysisResult.value = null
  uiAnalysisOpen.value = true
  uiAnalysisLoading.value = true
  try {
    uiAnalysisResult.value = await userInputAnalysis({ snapshot_ids: null })
  } catch (e) {
    message.error('用户输入分析失败：' + (e.message || ''))
    uiAnalysisOpen.value = false
  } finally {
    uiAnalysisLoading.value = false
  }
}

// ── Model analysis ───────────────────────────────────────────────────────────
async function openModelAnalysis() {
  modelAnalysisResult.value = null
  modelAnalysisOpen.value = true
  modelAnalysisLoading.value = true
  try {
    modelAnalysisResult.value = await getModelAnalysis(20)
  } catch (e) {
    message.error('模型分析失败：' + (e.message || ''))
    modelAnalysisOpen.value = false
  } finally {
    modelAnalysisLoading.value = false
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function fmtTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false }).replace(/\//g, '-')
}

function statusColor(s) {
  if (s === 'success') return 'green'
  if (s === 'failed') return 'red'
  if (s === 'max_iterations') return 'orange'
  return 'default'
}

function bcStatusColor(s) {
  if (s === 'pending') return 'orange'
  if (s === 'reviewed') return 'blue'
  if (s === 'fixed') return 'green'
  if (s === 'regressed') return 'red'
  return 'default'
}

function tcTypeColor(t) {
  if (t === 'golden') return 'gold'
  if (t === 'regression') return 'blue'
  if (t === 'calibration') return 'purple'
  return 'default'
}

function toolHeader(tc) {
  const errMark = tc.error ? ' ⚠' : ''
  const dur = tc.duration_ms != null ? ` · ${tc.duration_ms.toFixed(0)}ms` : ''
  return `#${tc.order}  ${tc.name}${errMark}${dur}`
}

function erStatusColor(s) {
  if (s === 'completed') return 'green'
  if (s === 'running') return 'blue'
  if (s === 'failed') return 'red'
  return 'orange'
}

function scoreClass(v) {
  if (v >= 0.7) return 'score-high'
  if (v >= 0.4) return 'score-mid'
  return 'score-low'
}

function progressColor(v) {
  if (v >= 0.7) return '#52c41a'
  if (v >= 0.4) return '#faad14'
  return '#ff4d4f'
}

// ── Init ───────────────────────────────────────────────────────────────────
function semCategoryColor(c) {
  const m = {
    retrieval_failure: 'blue', hallucination: 'red', tool_error: 'orange',
    reasoning_error: 'volcano', context_loss: 'purple',
    instruction_following: 'cyan', output_format: 'geekblue',
  }
  return m[c] || 'default'
}

function rootCauseColor(c) {
  const m = { prompt: 'blue', context: 'purple', tool: 'orange', model: 'gold', user_input: 'cyan' }
  return m[c] || 'default'
}

function confidenceCls(c) {
  if (c === 'high') return 'score-high'
  if (c === 'low') return 'score-low'
  return 'score-mid'
}

function confidenceTagColor(c) {
  if (c === 'high') return 'green'
  if (c === 'low') return 'red'
  return 'orange'
}

function proposalStatusColor(s) {
  if (s === 'approved') return 'green'
  if (s === 'rejected') return 'red'
  return 'orange'
}

function deltaCls(v) {
  if (v == null) return ''
  return v >= 0 ? 'score-high' : 'score-low'
}

function errorRateCls(v) {
  if (v > 0.2) return 'score-low'
  if (v > 0.05) return 'score-mid'
  return 'score-high'
}

function verdictColor(v) {
  if (v === 'transient') return 'green'
  if (v === 'kb_gap') return 'red'
  return 'default'
}

onMounted(() => {
  fetchStats()
  fetchSnaps()
  fetchBadcases()
  fetchTestCases()
  fetchEvalRunsList()
  loadCompletedRuns()
})
</script>

<style scoped>
.eval-page { padding: 24px 32px; max-width: 1200px; }

.page-header { display: flex; align-items: center; gap: 32px; margin-bottom: 20px; }
.page-header h2 { margin: 0; font-size: 20px; font-weight: 700; }

.stats-row { display: flex; gap: 16px; }
.stat-card {
  padding: 10px 20px;
  background: #f9f9f9;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  text-align: center;
  min-width: 80px;
}
.stat-card.warn { border-color: #faad14; background: #fffbe6; }
.stat-card.danger { border-color: #ff4d4f; background: #fff2f0; }
.stat-num { font-size: 22px; font-weight: 700; color: #333; line-height: 1.2; }
.stat-label { font-size: 11px; color: #888; margin-top: 2px; }

.filter-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.main-tabs { margin-top: -4px; }

.snap-table :deep(tr) { cursor: pointer; }
.snap-table :deep(tr:hover td) { background: #f5f5f5 !important; }

.token-hint { font-size: 11px; color: #888; }
.mono { font-family: monospace; font-size: 12px; }
.muted { color: #bbb; font-size: 12px; }

/* Drawer styles */
.detail-desc { margin-bottom: 16px; }
:deep(.ant-modal-body) { max-height: 75vh; overflow-y: auto; padding: 16px 20px; }
.section-title { font-size: 12px; font-weight: 600; color: #555; margin: 14px 0 6px; text-transform: uppercase; letter-spacing: .5px; }
.input-box {
  background: #f9f9f9;
  border: 1px solid #f0f0f0;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-all;
}
.response-box {
  background: #f6ffed;
  border: 1px solid #d9f7be;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 200px;
  overflow-y: auto;
}
.tool-chain :deep(.ant-collapse-header) { font-family: monospace; font-size: 12px; }
.tool-section-label { font-size: 11px; color: #888; margin-bottom: 4px; }
.code-block {
  background: #1e1e1e;
  color: #d4d4d4;
  border-radius: 4px;
  padding: 8px 10px;
  font-size: 11px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 160px;
  overflow-y: auto;
  margin: 0;
}
.code-block.result { background: #1a2b1a; color: #b5d6b5; }
.error-hint { color: #ff4d4f; font-size: 12px; margin-top: 4px; }
.tool-meta { font-size: 11px; color: #aaa; margin-top: 6px; }

.llm-calls { display: flex; flex-direction: column; gap: 4px; margin-bottom: 4px; }
.llm-call-row { display: flex; gap: 16px; font-size: 12px; padding: 4px 8px; background: #fafafa; border-radius: 4px; }
.llm-call-row .mono { color: #888; }

/* Score colors */
.score-high { color: #52c41a; font-weight: 600; }
.score-mid  { color: #faad14; }
.score-low  { color: #ff4d4f; }

/* Trends / eval runs */
.trends-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.dim-cards { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 4px; }
.dim-card {
  flex: 1;
  min-width: 160px;
  max-width: 220px;
  padding: 10px 14px;
  background: #fafafa;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
}
.dim-name { font-size: 12px; color: #555; margin-bottom: 6px; }
.dim-score { font-size: 18px; font-weight: 700; margin-top: 4px; text-align: right; }
.err-sample { font-size: 11px; color: #ff4d4f; background: #fff2f0; padding: 2px 6px; border-radius: 3px; margin-bottom: 2px; max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.parse-error-hint { margin-top: 16px; }
.parse-error-hint pre { margin: 8px 0 0; }

.field-hint { font-size: 11px; color: #aaa; }
</style>
