<template>
  <app-layout>
    <div class="kb-detail-page">
      <!-- Header -->
      <div class="page-header">
        <div class="header-left">
          <a-button type="text" @click="router.push('/knowledge')" style="padding: 0; margin-right: 8px">
            <arrow-left-outlined />
          </a-button>
          <h2 class="nr-serif">{{ kb?.name || '知识库' }}</h2>
          <a-tag color="green" v-if="kb?.status === 'active'" style="margin-left: 8px">活跃</a-tag>
        </div>
        <div class="header-right">
          <span class="stat-chip"><file-text-outlined /> {{ kb?.doc_count ?? 0 }} 篇</span>
          <span class="stat-chip"><database-outlined /> {{ kb?.chunk_count ?? 0 }} Chunk</span>
          <a-button @click="openEdit"><edit-outlined /> 设置</a-button>
        </div>
      </div>

      <a-tabs v-model:activeKey="activeTab">
        <!-- Tab 1: Documents -->
        <a-tab-pane key="docs" tab="文档">
          <div class="tab-toolbar">
            <a-tooltip :title="settingsStore.coverage.hasEmbedding ? '' : '缺 embedding provider，请到 Settings 添加'">
              <a-upload
                :show-upload-list="false"
                :before-upload="settingsStore.coverage.hasEmbedding ? openUploadModal : () => false"
                accept=".pdf,.md,.txt,.docx"
                :multiple="false"
                :disabled="!settingsStore.coverage.hasEmbedding"
              >
                <span :style="settingsStore.coverage.hasEmbedding ? {} : { display: 'inline-block', cursor: 'not-allowed' }">
                  <a-button type="primary" :disabled="!settingsStore.coverage.hasEmbedding"><upload-outlined /> 上传文档</a-button>
                </span>
              </a-upload>
            </a-tooltip>
          </div>

          <a-table
            :data-source="documents"
            :loading="docsLoading"
            :columns="docColumns"
            row-key="id"
            :pagination="false"
            size="middle"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'status'">
                <a-tag :color="statusColor(record.status)" size="small">{{ statusLabel(record.status) }}</a-tag>
                <a-spin v-if="isIndexing(record.status)" size="small" style="margin-left: 6px" />
              </template>
              <template v-if="column.key === 'file_size'">
                {{ formatSize(record.file_size) }}
              </template>
              <template v-if="column.key === 'action'">
                <a-button size="small" type="link" @click="openDocPreview(record)">预览</a-button>
                <a-button size="small" type="link" @click="downloadDoc(record)">下载</a-button>
                <a-button size="small" type="link" @click="openEntityModal(record)">实体</a-button>
                <a-popconfirm title="确定删除？" ok-type="danger" @confirm="removeDoc(record.id)">
                  <a-button size="small" type="link" danger>删除</a-button>
                </a-popconfirm>
              </template>
            </template>
          </a-table>
        </a-tab-pane>

        <!-- Tab 2: Chunk Browser -->
        <a-tab-pane key="chunks" tab="Chunk 浏览">
          <div class="chunk-browser">
            <div class="chunk-doc-list">
              <div
                v-for="doc in documents"
                :key="doc.id"
                class="chunk-doc-item"
                :class="{ active: selectedDocId === doc.id }"
                @click="selectDoc(doc)"
              >
                <file-text-outlined />
                <span class="chunk-doc-name">{{ doc.filename }}</span>
                <a-badge :count="doc.chunk_count" :overflow-count="999" color="var(--nr-clay)" />
              </div>
              <a-empty v-if="!documents.length" description="暂无文档" style="padding: 32px 0" />
            </div>

            <div class="chunk-list-panel">
              <a-spin :spinning="chunksLoading">
                <div v-if="docChunks.length" class="chunk-items">
                  <div
                    v-for="chunk in docChunks"
                    :key="chunk.id"
                    class="chunk-item"
                    @click="openChunkDetail(chunk)"
                  >
                    <div class="chunk-header">
                      <span class="chunk-index">#{{ chunk.chunk_index }}</span>
                      <a-tag
                        v-if="chunk.metadata?.content_type && chunk.metadata.content_type !== 'text'"
                        :color="contentTypeColor(chunk.metadata.content_type)"
                        size="small"
                        style="margin: 0"
                      >{{ chunk.metadata.content_type }}</a-tag>
                      <span class="chunk-tokens" v-if="chunk.token_count">{{ chunk.token_count }} tokens</span>
                    </div>
                    <div v-if="chunk.metadata?.section_path" class="chunk-section-path">
                      {{ chunk.metadata.section_path }}
                    </div>
                    <div class="chunk-preview">{{ chunk.content.slice(0, 120) }}{{ chunk.content.length > 120 ? '…' : '' }}</div>
                  </div>
                </div>
                <a-empty v-else-if="selectedDocId" description="该文档暂无 Chunk" />
                <div v-else class="chunk-placeholder">← 选择左侧文档查看 Chunk</div>
              </a-spin>
            </div>
          </div>
        </a-tab-pane>

        <!-- Tab 3: Test Query -->
        <a-tab-pane key="query" tab="测试检索">
          <div class="query-panel">
            <a-alert
              v-if="!settingsStore.coverage.hasEmbedding"
              type="warning"
              show-icon
              message="缺 embedding provider，请到 Settings 添加后才能使用检索功能。"
              style="margin-bottom: 12px"
            />
            <div class="query-input-row">
              <a-tooltip :title="settingsStore.coverage.hasEmbedding ? '' : '缺 embedding provider'">
                <a-input
                  v-model:value="queryText"
                  :placeholder="settingsStore.coverage.hasEmbedding ? '输入检索问题...' : '缺 embedding provider，请到 Settings 添加'"
                  :disabled="!settingsStore.coverage.hasEmbedding"
                  @pressEnter="runQuery"
                  style="flex: 1"
                />
              </a-tooltip>
              <a-input-number v-model:value="queryTopK" :min="1" :max="20" :step="1" style="width: 80px" />
              <a-segmented v-model:value="queryMode" :options="queryModeOptions" />
              <a-tooltip :title="settingsStore.coverage.hasEmbedding ? '' : '缺 embedding provider'">
                <span :style="settingsStore.coverage.hasEmbedding ? {} : { display: 'inline-block', cursor: 'not-allowed' }">
                  <a-button type="primary" :loading="queryLoading" :disabled="!settingsStore.coverage.hasEmbedding" @click="runQuery">检索</a-button>
                </span>
              </a-tooltip>
            </div>

            <a-spin :spinning="queryLoading">
              <div v-if="queryResults.length" class="query-results">
                <div
                  v-for="group in groupedQueryResults"
                  :key="group.filename"
                  class="result-file-group"
                >
                  <div class="result-file-header" @click="toggleGroup(group.filename)">
                    <file-text-outlined style="color: var(--nr-clay)" />
                    <span class="result-file-name">{{ group.filename }}</span>
                    <a-badge :count="group.chunks.length" color="var(--nr-clay)" :overflow-count="99" />
                    <caret-right-outlined
                      class="group-toggle-icon"
                      :class="{ expanded: !collapsedGroups.has(group.filename) }"
                    />
                  </div>
                  <div v-if="!collapsedGroups.has(group.filename)" class="result-file-chunks">
                    <div
                      v-for="r in group.chunks"
                      :key="r.chunk_id"
                      class="query-result-item"
                      @click="openResultDetail(r)"
                    >
                      <div class="result-header">
                        <span class="result-rank">#{{ r._rank }}</span>
                        <a-tooltip title="RRF 融合分">
                          <a-tag color="blue" size="small">{{ (r.score * 100).toFixed(2) }}%</a-tag>
                        </a-tooltip>
                        <a-tooltip v-if="r.dense_score != null" title="语义检索分">
                          <a-tag color="purple" size="small">D {{ (r.dense_score * 100).toFixed(1) }}%</a-tag>
                        </a-tooltip>
                        <a-tooltip v-if="r.sparse_score != null" title="关键词检索分">
                          <a-tag color="cyan" size="small">S {{ (r.sparse_score * 100).toFixed(1) }}%</a-tag>
                        </a-tooltip>
                        <a-tag v-if="r.dense_score == null" size="small" color="default">仅关键词</a-tag>
                        <a-tag v-if="r.sparse_score == null" size="small" color="default">仅语义</a-tag>
                      </div>
                      <div class="result-text">{{ r.text }}</div>
                    </div>
                  </div>
                </div>
              </div>
              <a-empty v-else-if="queryDone" description="未找到相关内容" />
            </a-spin>
          </div>
        </a-tab-pane>

        <!-- Tab 4: RAG Eval -->
        <a-tab-pane key="eval" tab="RAG 评估">
          <div class="eval-tab">
            <!-- RAGAS Model Config -->
            <a-collapse v-model:activeKey="ragasConfigOpen" ghost style="margin-bottom: 20px">
              <a-collapse-panel key="config" header="RAGAS 评估模型配置">
                <a-form layout="inline" style="gap: 12px; flex-wrap: wrap; align-items: flex-end">
                  <a-form-item label="Generator 模型" style="margin-bottom: 0">
                    <a-auto-complete
                      v-model:value="localRagasGen"
                      :options="settingsStore.allModelOptions"
                      placeholder="qwen-plus（默认）"
                      allow-clear
                      style="width: 200px"
                    />
                  </a-form-item>
                  <a-form-item label="Evaluator 模型" style="margin-bottom: 0">
                    <a-auto-complete
                      v-model:value="localRagasEval"
                      :options="settingsStore.allModelOptions"
                      placeholder="qwen-max（默认）"
                      allow-clear
                      style="width: 200px"
                    />
                  </a-form-item>
                  <a-form-item style="margin-bottom: 0">
                    <a-button type="primary" :loading="ragasSaving" @click="saveRagasSettings">保存配置</a-button>
                  </a-form-item>
                </a-form>
                <div style="font-size: 12px; color: var(--nr-ink-3); margin-top: 10px">
                  Generator：生成候选答案，推荐性价比高的模型；Evaluator：评估 Faithfulness / Recall，推荐能力强的模型
                </div>
              </a-collapse-panel>
            </a-collapse>

            <!-- Graph Expansion Toggle -->
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;padding:10px 12px;background:var(--nr-rail);border-radius:8px;border:1px solid var(--nr-border)">
              <a-switch
                :checked="kb?.enable_graph_expansion"
                :loading="graphToggleLoading"
                @change="toggleGraphExpansion"
              />
              <span style="font-size:13px;font-weight:500">图增强检索</span>
              <span style="font-size:12px;color:var(--nr-ink-3)">建图后自动开启，向量检索结果将追加跨文档实体邻居</span>
              <div style="flex:1"/>
              <a-button size="small" :loading="kbGraphBuilding" @click="triggerKbGraphBuild">
                {{ kbGraphBuilding ? '建图中…' : '重建知识图谱' }}
              </a-button>
              <a-button v-if="graphStats" size="small" @click="graphStatsVisible = true">查看图谱统计</a-button>
            </div>
            <!-- Graph Stats Modal -->
            <a-modal v-model:open="graphStatsVisible" title="知识图谱统计" :footer="null" width="480px">
              <div v-if="graphStats" style="padding:8px 0">
                <a-descriptions :column="2" size="small" bordered style="margin-bottom:16px">
                  <a-descriptions-item label="实体数">{{ graphStats.entity_count }}</a-descriptions-item>
                  <a-descriptions-item label="关系数">{{ graphStats.triple_count }}</a-descriptions-item>
                </a-descriptions>
                <div style="font-size:13px;font-weight:500;margin-bottom:8px">Top 实体（按提及次数）</div>
                <a-table
                  :data-source="graphStats.top_entities"
                  :columns="[{title:'实体',dataIndex:'name'},{title:'类型',dataIndex:'label'},{title:'提及次数',dataIndex:'mention_count',width:90}]"
                  size="small"
                  :pagination="false"
                  :scroll="{y:360}"
                  row-key="name"
                />
              </div>
            </a-modal>

            <!-- Eval Layout -->
            <div class="eval-layout">
              <!-- Datasets Panel -->
              <div class="datasets-panel">
                <div class="panel-header">
                  <h3>评估数据集</h3>
                  <div style="display:flex;gap:6px">
                    <a-button size="small" type="primary" @click="generateModalOpen = true"><thunderbolt-outlined /> 自动生成</a-button>
                    <a-upload :show-upload-list="false" :before-upload="handleDatasetUpload" accept=".jsonl">
                      <a-button size="small"><upload-outlined /> 上传 JSONL</a-button>
                    </a-upload>
                  </div>
                </div>

                <a-spin :spinning="datasetsLoading">
                  <div v-if="datasets.length" class="dataset-list">
                    <div
                      v-for="ds in datasets"
                      :key="ds.id"
                      class="dataset-item"
                      :class="{ active: selectedDataset?.id === ds.id }"
                      @click="selectedDataset = ds"
                    >
                      <div class="ds-name">{{ ds.name }}</div>
                      <div class="ds-meta">{{ ds.item_count }} 条 · {{ fmtDate(ds.created_at) }}</div>
                      <div class="ds-actions">
                        <a-button size="small" type="primary" @click.stop="startRun(ds)" :loading="startingRun === ds.id + ':quick'">Quick</a-button>
                        <a-button size="small" @click.stop="startRagasRun(ds)" :loading="startingRun === ds.id + ':ragas'">RAGAS</a-button>
                        <a-button size="small" @click.stop="startAgentRun(ds)" :loading="startingRun === ds.id + ':agent'">Agent</a-button>
                        <a-popconfirm title="确定删除数据集？" ok-type="danger" @confirm="removeDataset(ds.id)">
                          <a-button size="small" danger type="text" @click.stop><delete-outlined /></a-button>
                        </a-popconfirm>
                      </div>
                    </div>
                  </div>
                  <a-empty v-else description="暂无数据集" style="padding: 24px 0" />
                </a-spin>

                <div class="jsonl-hint">
                  JSONL 格式：每行 <code>{"query":"...","gold_answer":"...","gold_chunk_ids":[]}</code>
                </div>
              </div>

              <!-- Runs Panel -->
              <div class="runs-panel">
                <div class="panel-header">
                  <h3>评估历史</h3>
                  <a-button size="small" @click="loadRuns"><reload-outlined /> 刷新</a-button>
                </div>

                <a-spin :spinning="runsLoading">
                  <div v-if="evalRuns.length" class="runs-list">
                    <div
                      v-for="run in evalRuns"
                      :key="run.id"
                      class="run-card"
                      :class="{ expanded: expandedRun === run.id }"
                    >
                      <div class="run-header" @click="toggleRun(run)">
                        <div class="run-left">
                          <span class="run-name">{{ run.name }}</span>
                          <a-tag :color="run.eval_type === 'ragas' ? 'purple' : run.eval_type === 'agent' ? 'orange' : 'cyan'" size="small">{{ run.eval_type || 'quick' }}</a-tag>
                          <a-tag :color="runStatusColor(run.status)" size="small">{{ runStatusLabel(run.status) }}</a-tag>
                        </div>
                        <div class="run-right">
                          <span v-if="run.overall_score !== null" class="run-score">总分 {{ (run.overall_score * 100).toFixed(1) }}%</span>
                          <a-progress v-if="run.status === 'running'" :percent="runProgress(run)" size="small" style="width: 100px" />
                          <a-popconfirm title="确定删除？" ok-type="danger" @confirm.stop="removeRun(run)">
                            <a-button size="small" type="text" danger @click.stop><delete-outlined /></a-button>
                          </a-popconfirm>
                        </div>
                      </div>

                      <div v-if="run.metrics && Object.keys(run.metrics).length" class="run-metrics">
                        <span v-if="run.metrics._avg_hops !== undefined" class="metric-chip metric-hop">
                          平均跳数: {{ Number(run.metrics._avg_hops).toFixed(1) }}
                        </span>
                        <template v-for="(v, k) in run.metrics" :key="k">
                          <span v-if="!k.startsWith('_')" class="metric-chip">
                            {{ k }}: {{ (v * 100).toFixed(1) }}%
                          </span>
                        </template>
                      </div>

                      <div v-if="expandedRun === run.id" class="run-items">
                        <a-spin :spinning="itemsLoading">
                          <div v-if="runItems.length" class="items-table">
                            <div class="items-header">
                              <span style="flex: 2">问题</span>
                              <span style="flex: 2">参考答案</span>
                              <span v-if="expandedRunType === 'ragas' || expandedRunType === 'agent'" style="flex: 2">生成答案</span>
                              <span v-if="expandedRunType === 'agent'" style="width: 48px; text-align: center">跳数</span>
                              <span v-if="expandedRunType === 'agent'" style="flex: 2">工具链</span>
                              <span v-for="k in metricKeys" :key="k" style="width: 80px; text-align: center">{{ k }}</span>
                            </div>
                            <div v-for="item in runItems" :key="item.id" class="items-row">
                              <span style="flex: 2" class="item-text">{{ item.query }}</span>
                              <span style="flex: 2" class="item-text">{{ item.gold_answer || '-' }}</span>
                              <span v-if="expandedRunType === 'ragas' || expandedRunType === 'agent'" style="flex: 2" class="item-text">{{ item.generated_answer || '-' }}</span>
                              <span v-if="expandedRunType === 'agent'" style="width: 48px; text-align: center; font-size: 12px">
                                {{ item.metrics?._hops ?? '-' }}
                              </span>
                              <span v-if="expandedRunType === 'agent'" style="flex: 2; font-size: 11px; color: var(--nr-ink-2)" class="item-text">
                                {{ (item.metrics?._tools_used || []).join(' → ') || '-' }}
                              </span>
                              <span
                                v-for="k in metricKeys" :key="k"
                                style="width: 80px; text-align: center; font-size: 12px"
                                :class="metricColor(item.metrics?.[k])"
                              >
                                {{ item.metrics?.[k] !== undefined ? (item.metrics[k] * 100).toFixed(0) + '%' : '-' }}
                              </span>
                            </div>
                          </div>
                          <a-empty v-else description="暂无逐条结果" style="padding: 16px 0" />
                        </a-spin>
                      </div>
                    </div>
                  </div>
                  <a-empty v-else description="暂无评估运行" style="padding: 40px 0" />
                </a-spin>
              </div>
            </div>
          </div>
        </a-tab-pane>

        <!-- Tab 5: Knowledge Graph / Wiki -->
        <a-tab-pane key="graph" tab="知识图谱/Wiki">
          <div class="wiki-wrap">
            <div class="wiki-list">
              <a-input-search v-model:value="wikiSearch" placeholder="搜索实体…"
                allow-clear @search="loadWikiEntities" style="margin-bottom:8px" />
              <a-spin :spinning="wikiLoading">
                <template v-if="wikiLoading" />
                <template v-else-if="!wikiEntities.length && wikiSearch">
                  <a-empty description="无匹配实体" />
                </template>
                <template v-else-if="!wikiEntities.length">
                  <a-empty description="暂无知识图谱数据，请先在「文档」页重建知识图谱" />
                </template>
                <template v-else>
                  <div v-for="e in wikiEntities" :key="e.name"
                    class="wiki-ent" :class="{ active: entityDetail?.name === e.name }"
                    @click="selectEntity(e.name)">
                    <span class="wiki-ent-name">{{ e.name }}</span>
                    <span class="wiki-ent-count">{{ e.mentions }}</span>
                  </div>
                </template>
              </a-spin>
            </div>
            <div class="wiki-detail">
              <a-spin :spinning="detailLoading">
                <template v-if="entityDetail">
                  <h2 class="wiki-title">{{ entityDetail.name }}
                    <a-tag>{{ entityDetail.label }}</a-tag>
                    <span class="wiki-sub">被 {{ entityDetail.mention_count }} 处提及</span>
                  </h2>

                  <div class="wiki-article">
                    <a-spin :spinning="articleLoading">
                      <template v-if="article">
                        <CitationText :text="article.markdown" :citations="article.citations" />
                        <div class="wiki-article-meta">
                          <a-tag v-if="article.stale" color="orange">来源已更新</a-tag>
                          <a-button size="small" type="link" @click="genArticle">
                            {{ article.stale ? '重新生成' : '重新生成词条' }}
                          </a-button>
                        </div>
                      </template>
                      <a-button v-else type="dashed" block @click="genArticle">生成词条</a-button>
                    </a-spin>
                  </div>

                  <h3>事实</h3>
                  <a-empty v-if="!entityDetail.facts.length" description="无事实" />
                  <div v-for="f in entityDetail.facts" :key="f.triple_id" class="wiki-fact">
                    <div class="wiki-fact-row" @click="toggleTripleEvidence(f.triple_id)">
                      <span>{{ f.source }} <em>—{{ f.label }}→</em> {{ f.target }}</span>
                      <a-tag color="blue">{{ f.doc_count }} 篇文档</a-tag>
                    </div>
                    <div v-if="expandedTriple === f.triple_id" class="wiki-evidence">
                      <div v-for="(c, i) in tripleChunks" :key="i" class="wiki-chunk">
                        <div class="wiki-chunk-src">{{ c.source }}<span v-if="c.page != null"> · p{{ c.page }}</span></div>
                        <div class="wiki-chunk-text">{{ c.content }}</div>
                      </div>
                    </div>
                  </div>

                  <h3 style="margin-top:16px">邻居实体</h3>
                  <a-tag v-for="n in entityDetail.neighbors" :key="n"
                    class="wiki-neighbor" @click="selectEntity(n)">{{ n }}</a-tag>

                  <a-collapse ghost style="margin-top:8px">
                    <a-collapse-panel key="graph" header="知识图谱">
                      <EntityNeighborGraph
                        :center="entityDetail.name"
                        :neighbors="entityDetail.neighbors"
                        @select="selectEntity" />
                    </a-collapse-panel>
                  </a-collapse>
                </template>
                <a-empty v-else description="选择左侧实体查看详情" />
              </a-spin>
            </div>
          </div>
        </a-tab-pane>

      </a-tabs>

      <!-- Query result detail modal -->
      <a-modal v-model:open="resultDetailOpen" title="检索结果详情" :footer="null" width="760">
        <div v-if="selectedResult">
          <div class="chunk-meta-pills">
            <span class="meta-pill">排名 #{{ selectedResult._rank }}</span>
            <a-tooltip title="RRF 融合分">
              <a-tag color="blue">{{ (selectedResult.score * 100).toFixed(2) }}%</a-tag>
            </a-tooltip>
            <a-tooltip v-if="selectedResult.dense_score != null" title="语义检索分">
              <a-tag color="purple">D {{ (selectedResult.dense_score * 100).toFixed(1) }}%</a-tag>
            </a-tooltip>
            <a-tooltip v-if="selectedResult.sparse_score != null" title="关键词检索分">
              <a-tag color="cyan">S {{ (selectedResult.sparse_score * 100).toFixed(1) }}%</a-tag>
            </a-tooltip>
          </div>
          <div class="chunk-content markdown-body" v-html="renderMarkdown(selectedResult.text)"></div>
          <template v-if="selectedResult.metadata && Object.keys(selectedResult.metadata).length">
            <div class="chunk-meta-title">元数据</div>
            <div class="chunk-meta-structured">
              <template v-for="(val, key) in structuredMetaFields(selectedResult.metadata)" :key="key">
                <div class="meta-row">
                  <span class="meta-key">{{ key }}</span>
                  <span class="meta-val">{{ val }}</span>
                </div>
              </template>
              <template v-if="Object.keys(rawMetaRemainder(selectedResult.metadata)).length">
                <div class="meta-row meta-row-json">
                  <span class="meta-key">其他</span>
                  <pre class="meta-val-json">{{ JSON.stringify(rawMetaRemainder(selectedResult.metadata), null, 2) }}</pre>
                </div>
              </template>
            </div>
          </template>
        </div>
      </a-modal>

      <!-- KB settings modal -->
      <a-modal
        v-model:open="editOpen"
        title="知识库设置"
        ok-text="保存"
        cancel-text="取消"
        :confirm-loading="editSaving"
        @ok="saveEdit"
        width="440"
      >
        <a-form layout="vertical" style="margin-top: 16px">
          <a-form-item label="名称" required>
            <a-input v-model:value="editForm.name" placeholder="知识库名称" />
          </a-form-item>
          <a-form-item label="描述">
            <a-textarea v-model:value="editForm.description" :rows="3" placeholder="可选" />
          </a-form-item>
        </a-form>
      </a-modal>

      <!-- Upload options modal -->
      <a-modal
        v-model:open="uploadModalOpen"
        title="上传文档"
        ok-text="上传"
        cancel-text="取消"
        :confirm-loading="uploadingFile"
        @ok="confirmUpload"
        @cancel="pendingFile = null"
        width="400"
      >
        <a-form layout="vertical" style="margin-top: 16px">
          <a-form-item label="文件">
            <span style="color: var(--nr-ink)">{{ pendingFile?.name }}</span>
          </a-form-item>
          <a-form-item label="PDF 解析器" v-if="isPdf(pendingFile)">
            <a-select v-model:value="uploadPdfParser" style="width: 100%">
              <a-select-option value="mineru">MinerU（学术文档 OCR，推荐）</a-select-option>
              <a-select-option value="marker">Marker（高质量 OCR）</a-select-option>
              <a-select-option value="markitdown">MarkItDown（快速，无 OCR）</a-select-option>
            </a-select>
          </a-form-item>
        </a-form>
      </a-modal>

      <!-- Unified document preview modal -->
      <a-modal
        v-model:open="previewOpen"
        :footer="null"
        :closable="false"
        :width="previewViewMode === 'source' ? '1200px' : '900px'"
        :body-style="{ height: '80vh', padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden' }"
        @after-close="closePreview"
      >
        <template #title>
          <div class="pdf-modal-title">
            <span class="pdf-modal-filename">{{ previewDoc?.filename }}</span>
            <span class="pdf-modal-count" v-if="previewChunks.length">{{ previewChunks.length }} 个 Chunk</span>
            <a-segmented
              v-model:value="previewViewMode"
              :options="previewViewOptions"
              size="small"
              style="margin-left: 16px"
            />
            <a-button type="text" size="small" style="margin-left: auto" @click="previewOpen = false">
              <close-outlined />
            </a-button>
          </div>
        </template>

        <div v-if="previewLoading" class="pdf-loading">
          <a-spin tip="加载中..." />
        </div>

        <!-- 源文件视图 -->
        <div v-else-if="previewViewMode === 'source'" class="pdf-viewer-body">
          <div class="pdf-left">
            <template v-if="pdfBlobUrl">
              <vue-pdf-embed
                ref="pdfRef"
                :source="pdfBlobUrl"
                :page="currentPdfPage"
                :text-layer="true"
                class="pdf-embed"
                @loaded="onPdfLoaded"
              />
              <div class="pdf-nav">
                <a-button size="small" :disabled="currentPdfPage <= 1" @click="currentPdfPage--">‹</a-button>
                <span class="pdf-nav-label">{{ currentPdfPage }} / {{ totalPdfPages }}</span>
                <a-button size="small" :disabled="currentPdfPage >= totalPdfPages" @click="currentPdfPage++">›</a-button>
              </div>
            </template>
            <div v-else class="pdf-error">
              <file-text-outlined style="font-size: 32px; color: var(--nr-ink-3)" />
              <div style="margin-top: 8px; color: var(--nr-ink-3)">源文件不存在，请重新上传文档以启用预览</div>
            </div>
          </div>
          <div class="pdf-right">
            <div class="pdf-chunk-list">
              <div
                v-for="chunk in previewChunks"
                :key="chunk.id"
                class="pdf-chunk-card"
                :class="{ active: selectedPreviewChunkId === chunk.id }"
                @click="selectPreviewChunk(chunk.id)"
              >
                <div class="pdf-chunk-header">
                  <span class="pdf-chunk-index">#{{ chunk.chunk_index }}</span>
                  <a-tag v-if="chunk.metadata?.page_num ?? chunk.metadata?.page" size="small" color="blue">
                    第 {{ chunk.metadata.page_num ?? chunk.metadata.page }} 页
                  </a-tag>
                  <span class="pdf-chunk-tokens" v-if="chunk.token_count">{{ chunk.token_count }} tokens</span>
                </div>
                <div class="pdf-chunk-preview">{{ chunk.content.slice(0, 80).replace(/\n+/g, ' ') }}{{ chunk.content.length > 80 ? '…' : '' }}</div>
              </div>
              <a-empty v-if="!previewChunks.length" description="暂无 Chunk" />
            </div>
          </div>
        </div>

        <!-- Markdown 视图 -->
        <div v-else-if="previewViewMode === 'markdown'" class="fulltext-body markdown-body" v-html="previewFullTextHtml" />

        <!-- Chunks 视图 -->
        <div v-else-if="previewViewMode === 'chunks'" class="chunks-grid-panel">
          <div class="chunks-grid">
            <div v-for="chunk in previewChunks" :key="chunk.id" class="chunks-grid-card" @click="openChunkDetail(chunk)">
              <div class="chunks-grid-header">
                <span class="pdf-chunk-index">#{{ chunk.chunk_index }}</span>
                <a-tag v-if="chunk.metadata?.page_num ?? chunk.metadata?.page" size="small" color="blue">
                  第 {{ chunk.metadata.page_num ?? chunk.metadata.page }} 页
                </a-tag>
                <a-tag v-if="chunk.metadata?.content_type && chunk.metadata.content_type !== 'text'"
                  :color="contentTypeColor(chunk.metadata.content_type)" size="small">
                  {{ chunk.metadata.content_type }}
                </a-tag>
                <span class="pdf-chunk-tokens" v-if="chunk.token_count">{{ chunk.token_count }} tokens</span>
              </div>
              <div class="chunks-grid-content">{{ chunk.content.replace(/\n+/g, ' ') }}</div>
            </div>
          </div>
          <a-empty v-if="!previewChunks.length" description="暂无 Chunk" style="padding: 40px 0" />
        </div>
      </a-modal>

      <!-- Chunk detail modal -->
      <a-modal v-model:open="chunkModalOpen" title="Chunk 详情" :footer="null" width="720">
        <div v-if="selectedChunk">
          <div class="chunk-meta-pills">
            <span class="meta-pill">#{{ selectedChunk.chunk_index }}</span>
            <span class="meta-pill" v-if="selectedChunk.token_count">{{ selectedChunk.token_count }} tokens</span>
            <span class="meta-pill" v-if="selectedChunk.char_start != null">字符 {{ selectedChunk.char_start }}–{{ selectedChunk.char_end }}</span>
            <a-tag
              v-if="selectedChunk.metadata?.content_type"
              :color="contentTypeColor(selectedChunk.metadata.content_type)"
              size="small"
            >{{ selectedChunk.metadata.content_type }}</a-tag>
          </div>
          <div class="chunk-content markdown-body" v-html="renderMarkdown(selectedChunk.content)"></div>
          <template v-if="selectedChunk.metadata && Object.keys(selectedChunk.metadata).length">
            <div class="chunk-meta-title">元数据</div>
            <div class="chunk-meta-structured">
              <template v-for="(val, key) in structuredMetaFields(selectedChunk.metadata)" :key="key">
                <div class="meta-row">
                  <span class="meta-key">{{ key }}</span>
                  <span class="meta-val">{{ val }}</span>
                </div>
              </template>
              <template v-if="Object.keys(rawMetaRemainder(selectedChunk.metadata)).length">
                <div class="meta-row meta-row-json">
                  <span class="meta-key">其他</span>
                  <pre class="meta-val-json">{{ JSON.stringify(rawMetaRemainder(selectedChunk.metadata), null, 2) }}</pre>
                </div>
              </template>
            </div>
          </template>
        </div>
      </a-modal>

      <!-- Entity extraction modal -->
      <a-modal
        v-model:open="entityModalOpen"
        :title="entityModalDoc ? `实体抽取 — ${entityModalDoc.filename}` : '实体抽取'"
        :footer="null"
        width="680px"
        @cancel="_stopEntityPoll"
      >
        <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
          <a-button type="primary" size="small" :loading="entityBuilding.has(entityModalDoc?.id)" @click="triggerDocGraphBuild">
            {{ entityBuilding.has(entityModalDoc?.id) ? '抽取中…' : '重新抽取' }}
          </a-button>
          <span style="color: var(--nr-ink-3); font-size: 12px;">首次打开会自动触发抽取，完成后刷新查看结果</span>
        </div>
        <a-spin :spinning="entityLoading">
          <div v-if="entityList.length === 0 && !entityLoading" style="color: var(--nr-ink-3); padding: 24px 0; text-align: center;">
            暂无实体数据，点击「重新抽取」生成
          </div>
          <div v-else style="display: flex; flex-wrap: wrap; gap: 6px; max-height: 400px; overflow-y: auto; padding: 4px 0;">
            <a-tag
              v-for="e in entityList"
              :key="e.name + e.label"
              :color="entityLabelColor(e.label)"
            >
              {{ e.name }}
              <span style="opacity: 0.65; font-size: 11px; margin-left: 3px;">({{ e.label }}) ×{{ e.mentions }}</span>
            </a-tag>
          </div>
        </a-spin>
      </a-modal>

      <!-- Generate Dataset Modal -->
      <a-modal
        v-model:open="generateModalOpen"
        title="自动生成评估数据集"
        ok-text="开始生成"
        cancel-text="取消"
        :confirm-loading="generatingDataset"
        @ok="submitGenerateDataset"
        width="440"
      >
        <a-form layout="vertical" style="margin-top: 16px">
          <a-form-item label="数据集名称">
            <a-input v-model:value="generateForm.name" placeholder="留空自动命名" />
          </a-form-item>
          <a-form-item label="题目数量">
            <a-input-number v-model:value="generateForm.n_questions" :min="5" :max="100" style="width:100%" />
          </a-form-item>
          <a-form-item label="单跳题比例">
            <a-slider v-model:value="generateForm.simple_ratio" :min="0" :max="1" :step="0.1" :marks="{0:'0',0.5:'50%',1:'100%'}" />
          </a-form-item>
          <a-form-item label="多跳题比例">
            <a-slider v-model:value="generateForm.multi_context_ratio" :min="0" :max="1" :step="0.1" :marks="{0:'0',0.5:'50%',1:'100%'}" />
          </a-form-item>
          <div style="font-size:12px;color:var(--nr-ink-3)">推理题比例 = 1 − 单跳 − 多跳；由 RAGAS TestsetGenerator 自动生成，需要知识库已有文档</div>
        </a-form>
      </a-modal>
    </div>
  </app-layout>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, reactive, watch } from 'vue'
import VuePdfEmbed from 'vue-pdf-embed'
import 'vue-pdf-embed/dist/styles/textLayer.css'
import 'vue-pdf-embed/dist/styles/annotationLayer.css'
import { marked } from 'marked'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined, UploadOutlined, FileTextOutlined, DatabaseOutlined,
  CaretRightOutlined, DeleteOutlined, ReloadOutlined, EditOutlined, CloseOutlined, ThunderboltOutlined
} from '@ant-design/icons-vue'
import AppLayout from '@/layouts/AppLayout.vue'
import EntityNeighborGraph from '@/components/EntityNeighborGraph.vue'
import CitationText from '@/components/CitationText.vue'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useSettingsStore } from '@/stores/settings'
import { apiPost } from '@/apis/base'
import { listDocumentChunks, testQuery, getDocumentFileBlob, buildDocGraph, getDocEntities, buildKbGraph, getGraphStats, listGraphEntities, getGraphEntity, getTripleChunks, getEntityArticle, generateEntityArticle } from '@/apis/knowledge'
import {
  listDatasets, uploadDataset, deleteDataset, generateDataset,
  listEvalRuns, createEvalRun, createRagasRun, createAgentRun, getEvalRun, deleteEvalRun
} from '@/apis/knowledge'

const route = useRoute()
const router = useRouter()
const kbStore = useKnowledgeStore()
const settingsStore = useSettingsStore()

const kbId = route.params.id
const kb = computed(() => kbStore.current)

const activeTab = ref('docs')

// ── Documents ──
const documents = ref([])
const docsLoading = ref(false)
const selectedDocId = ref(null)
const docChunks = ref([])
const chunksLoading = ref(false)

// ── Unified preview modal ──
const previewOpen = ref(false)
const previewDoc = ref(null)
const previewChunks = ref([])
const previewLoading = ref(false)
const previewViewMode = ref('markdown')
const pdfBlobUrl = ref('')
const selectedPreviewChunkId = ref(null)
const pdfRef = ref(null)
const currentPdfPage = ref(1)
const totalPdfPages = ref(0)

const isPdfDoc = computed(() => previewDoc.value?.filename?.toLowerCase().endsWith('.pdf'))

const previewViewOptions = computed(() => {
  const opts = []
  if (isPdfDoc.value) opts.push({ label: '源文件', value: 'source' })
  opts.push({ label: 'Markdown', value: 'markdown' })
  opts.push({ label: 'Chunks', value: 'chunks' })
  return opts
})

const previewImageTicket = ref('')

function rewriteLocalImagePaths(md, ticket) {
  if (!ticket) return md
  return md.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
    const decoded = decodeURIComponent(src)
    const isLocal = /^[A-Za-z]:[\\/]/.test(decoded) || /^\/(?!\/)[^/]/.test(decoded)
    if (!isLocal) return match
    return `![${alt}](/api/rag/images?path=${encodeURIComponent(decoded)}&ticket=${encodeURIComponent(ticket)})`
  })
}

const previewFullTextHtml = computed(() => {
  if (!previewChunks.value.length) return ''
  const sorted = [...previewChunks.value].sort((a, b) => a.chunk_index - b.chunk_index)
  const raw = sorted.map(c => c.content).join('\n\n---\n\n')
  return marked.parse(rewriteLocalImagePaths(raw, previewImageTicket.value))
})

async function openDocPreview(doc) {
  previewDoc.value = doc
  previewOpen.value = true
  previewLoading.value = true
  previewChunks.value = []
  pdfBlobUrl.value = ''
  previewImageTicket.value = ''
  selectedPreviewChunkId.value = null
  previewViewMode.value = doc.filename?.toLowerCase().endsWith('.pdf') ? 'source' : 'markdown'
  try {
    const isPdf = doc.filename?.toLowerCase().endsWith('.pdf')
    const [chunks, ticketRes] = await Promise.all([
      listDocumentChunks(kbId, doc.id),
      apiPost('/api/rag/image-ticket', {}),
    ])
    previewChunks.value = chunks
    previewImageTicket.value = ticketRes.ticket || ''
    if (isPdf) {
      try {
        const blob = await getDocumentFileBlob(kbId, doc.id)
        pdfBlobUrl.value = URL.createObjectURL(blob)
      } catch (_) {
        // 源文件不可用，模板已有"请重新上传"提示
      }
    }
  } catch (e) {
    message.error('加载预览失败：' + (e.message || ''))
  } finally {
    previewLoading.value = false
  }
}

function selectPreviewChunk(id) {
  selectedPreviewChunkId.value = id
  const chunk = previewChunks.value.find(c => c.id === id)
  const pageNum = chunk?.metadata?.page_num ?? chunk?.metadata?.page
  if (pageNum) currentPdfPage.value = pageNum
}

function onPdfLoaded(pdf) {
  totalPdfPages.value = pdf.numPages
}

function closePreview() {
  if (pdfBlobUrl.value) {
    URL.revokeObjectURL(pdfBlobUrl.value)
    pdfBlobUrl.value = ''
  }
  previewDoc.value = null
  previewChunks.value = []
  selectedPreviewChunkId.value = null
  currentPdfPage.value = 1
  totalPdfPages.value = 0
}

async function downloadDoc(doc) {
  try {
    const blob = await getDocumentFileBlob(kbId, doc.id)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = doc.filename
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    message.error('下载失败：' + (e.message || ''))
  }
}

// ── Entity extraction modal ──
const entityModalOpen = ref(false)
const entityModalDoc = ref(null)
const entityList = ref([])
const entityLoading = ref(false)
const entityBuilding = ref(new Set())
const _entityPollTimers = {}  // docId → intervalId

const ENTITY_LABEL_COLORS = {
  Method: 'blue', Model: 'geekblue', Dataset: 'cyan', Concept: 'purple',
  Organization: 'orange', Person: 'gold', Term: 'green',
}
function entityLabelColor(label) {
  return ENTITY_LABEL_COLORS[label] || 'default'
}

function _stopEntityPoll(docId) {
  if (docId) {
    if (_entityPollTimers[docId]) { clearInterval(_entityPollTimers[docId]); delete _entityPollTimers[docId] }
  } else {
    Object.keys(_entityPollTimers).forEach(id => { clearInterval(_entityPollTimers[id]); delete _entityPollTimers[id] })
  }
}

async function openEntityModal(doc) {
  entityModalDoc.value = doc
  entityList.value = []
  entityModalOpen.value = true
  await loadDocEntities(doc)
}

async function loadDocEntities(doc) {
  entityLoading.value = true
  try {
    const res = await getDocEntities(kbId, doc.id)
    entityList.value = res.entities || []
  } catch {
    entityList.value = []
  } finally {
    entityLoading.value = false
  }
}

async function triggerDocGraphBuild() {
  if (!entityModalDoc.value) return
  const docId = entityModalDoc.value.id
  entityBuilding.value = new Set([...entityBuilding.value, docId])
  _stopEntityPoll(docId)
  try {
    await buildDocGraph(kbId, docId)
    message.info('抽取任务已提交，正在后台运行…')
    const clearBuilding = (id) => {
      const next = new Set(entityBuilding.value); next.delete(id); entityBuilding.value = next
    }
    let attempts = 0
    _entityPollTimers[docId] = setInterval(async () => {
      attempts++
      try {
        const res = await getDocEntities(kbId, docId)
        const list = res.entities || []
        if (list.length > 0) {
          if (entityModalDoc.value?.id === docId) entityList.value = list
          clearBuilding(docId)
          _stopEntityPoll(docId)
          message.success(`抽取完成，共 ${list.length} 个实体`)
        }
      } catch { /* ignore poll errors */ }
      if (attempts >= 36) {
        clearBuilding(docId)
        _stopEntityPoll(docId)
      }
    }, 5000)
  } catch (e) {
    message.error('抽取失败：' + (e.message || ''))
    const next = new Set(entityBuilding.value); next.delete(docId); entityBuilding.value = next
  }
}

// ── Upload modal ──
const uploadModalOpen = ref(false)
const pendingFile = ref(null)
const uploadPdfParser = ref('mineru')
const uploadingFile = ref(false)

function isPdf(file) {
  if (!file) return false
  return file.name?.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf'
}

function openUploadModal(file) {
  pendingFile.value = file
  uploadPdfParser.value = 'marker'
  uploadModalOpen.value = true
  return false  // prevent ant-design auto upload
}

async function confirmUpload() {
  if (!pendingFile.value) return
  uploadingFile.value = true
  try {
    const parser = isPdf(pendingFile.value) ? uploadPdfParser.value : 'markitdown'
    await kbStore.uploadDoc(kbId, pendingFile.value, parser)
    await loadDocs()
    message.success(`${pendingFile.value.name} 上传成功，正在解析…`)
    uploadModalOpen.value = false
    pendingFile.value = null
  } catch (e) {
    message.error(e.message || '上传失败')
  } finally {
    uploadingFile.value = false
  }
}

// ── Query ──
const queryText = ref('')
const queryTopK = ref(5)
const queryMode = ref('hybrid')
const queryModeOptions = [
  { label: '混合', value: 'hybrid' },
  { label: '语义', value: 'dense' },
  { label: '关键词', value: 'sparse' },
]
const queryLoading = ref(false)
const queryResults = ref([])
const queryDone = ref(false)
const chunkModalOpen = ref(false)
const selectedChunk = ref(null)
const resultDetailOpen = ref(false)
const selectedResult = ref(null)
const collapsedGroups = reactive(new Set())

// ── Edit KB ──
const editOpen = ref(false)
const editSaving = ref(false)
const editForm = reactive({ name: '', description: '' })

function openEdit() {
  editForm.name = kb.value?.name || ''
  editForm.description = kb.value?.description || ''
  editOpen.value = true
}

async function saveEdit() {
  if (!editForm.name.trim()) { message.warning('请填写名称'); return }
  editSaving.value = true
  try {
    await kbStore.update(kbId, { name: editForm.name, description: editForm.description || null })
    editOpen.value = false
    message.success('已保存')
  } catch (e) {
    message.error(e.message || '保存失败')
  } finally {
    editSaving.value = false
  }
}

// ── Eval ──
const ragasConfigOpen = ref([])
const localRagasGen  = ref(null)
const localRagasEval = ref(null)
const ragasSaving    = ref(false)

const graphToggleLoading = ref(false)
const kbGraphBuilding = ref(false)
const graphStats = ref(null)
const graphStatsVisible = ref(false)

const datasets = ref([])
const datasetsLoading = ref(false)
const selectedDataset = ref(null)
const generateModalOpen = ref(false)
const generatingDataset = ref(false)
const generateForm = ref({ name: '', n_questions: 20, simple_ratio: 0.3, multi_context_ratio: 0.5 })
const evalRuns = ref([])
const runsLoading = ref(false)
const startingRun = ref(null)
const expandedRun = ref(null)
const expandedRunType = ref('quick')
const runItems = ref([])
const itemsLoading = ref(false)

const metricKeys = computed(() => {
  const keys = new Set()
  runItems.value.forEach(i => Object.keys(i.metrics || {}).forEach(k => {
    if (!k.startsWith('_')) keys.add(k)
  }))
  return [...keys]
})

let pollTimer = null
let evalPollTimer = null

const docColumns = [
  {
    title: '文件名', dataIndex: 'filename', key: 'filename',
    sorter: (a, b) => (a.filename || '').localeCompare(b.filename || '', 'zh-Hans-CN'),
  },
  { title: '大小', key: 'file_size', width: 100 },
  { title: 'Chunk 数', dataIndex: 'chunk_count', key: 'chunk_count', width: 90 },
  { title: '图片数', dataIndex: 'image_count', key: 'image_count', width: 80 },
  { title: '状态', key: 'status', width: 110 },
  { title: '操作', key: 'action', width: 180 },
]

onMounted(async () => {
  await kbStore.fetchOne(kbId)
  await settingsStore.fetchAll()
  syncRagas()
  await loadDocs()
  startPoll()
  // Load eval data when tab first activated
  if (activeTab.value === 'eval') await loadEvalData()
})

onUnmounted(() => {
  stopPoll()
  stopEvalPoll()
})

// ── Wiki / Knowledge Graph entity browser ──
const wikiEntities   = ref([])        // [{name,label,mentions}]
const wikiSearch     = ref('')
const wikiLoading    = ref(false)
const entityDetail   = ref(null)      // {name,label,mention_count,facts,neighbors}
const detailLoading  = ref(false)
const expandedTriple = ref(null)      // triple_id whose evidence is open
const tripleChunks   = ref([])        // evidence chunks for expandedTriple

// ── LLM entity article ──
const article        = ref(null)      // {markdown, citations, model, generated_at, stale}
const articleLoading = ref(false)

async function loadArticle(name) {
  article.value = null
  try { const r = await getEntityArticle(kbId, name); article.value = r.article } catch (e) {}
}
async function genArticle() {
  if (!entityDetail.value) return
  articleLoading.value = true
  try {
    const r = await generateEntityArticle(kbId, entityDetail.value.name)
    article.value = r.article
  } catch (e) { message.error('生成词条失败') }
  finally { articleLoading.value = false }
}

async function loadWikiEntities() {
  wikiLoading.value = true
  try {
    const r = await listGraphEntities(kbId, { search: wikiSearch.value, limit: 100 })
    wikiEntities.value = r.entities || []
  } catch (e) {
    message.error('加载实体失败')
  } finally { wikiLoading.value = false }
}

async function selectEntity(name) {
  detailLoading.value = true
  expandedTriple.value = null
  tripleChunks.value = []
  try {
    entityDetail.value = await getGraphEntity(kbId, name)
    await loadArticle(name)
  } catch (e) {
    message.error('加载实体详情失败')
  } finally { detailLoading.value = false }
}

async function toggleTripleEvidence(tripleId) {
  if (expandedTriple.value === tripleId) { expandedTriple.value = null; return }
  expandedTriple.value = tripleId
  tripleChunks.value = []
  try {
    const r = await getTripleChunks(kbId, tripleId)
    tripleChunks.value = r.chunks || []
  } catch (e) {
    message.error('加载证据失败')
  }
}

// Load eval data when switching to eval tab
watch(activeTab, async (tab) => {
  if (tab === 'eval' && !datasets.value.length && !evalRuns.value.length) {
    await loadEvalData()
    startEvalPoll()
  }
  if (tab === 'graph' && !wikiEntities.value.length) {
    await loadWikiEntities()
  }
})

async function loadDocs() {
  docsLoading.value = true
  try { documents.value = await kbStore.fetchDocuments(kbId) }
  finally { docsLoading.value = false }
}

function startPoll() {
  pollTimer = setInterval(() => {
    if (documents.value.some(d => isIndexing(d.status))) loadDocs()
  }, 3000)
}

function stopPoll() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}


async function triggerKbGraphBuild() {
  kbGraphBuilding.value = true
  try {
    await buildKbGraph(kbId)
    message.success('建图任务已启动，完成后图增强检索将自动开启')
    let attempts = 0
    const timer = setInterval(async () => {
      attempts++
      try {
        const stats = await getGraphStats(kbId)
        if (stats.entity_count > 0) {
          graphStats.value = stats
          await kbStore.fetchOne(kbId)
          kbGraphBuilding.value = false
          clearInterval(timer)
          message.success(`建图完成，共 ${stats.entity_count} 个实体`)
        }
      } catch { /* ignore */ }
      if (attempts >= 36) { // 6 min timeout
        clearInterval(timer)
        kbGraphBuilding.value = false
      }
    }, 10000)
  } catch (e) {
    message.error(e.message || '建图失败')
    kbGraphBuilding.value = false
  }
}

async function toggleGraphExpansion(val) {
  graphToggleLoading.value = true
  try {
    await kbStore.update(kbId, { enable_graph_expansion: val })
    await kbStore.fetchOne(kbId)
  } catch (e) {
    message.error(e.message || '更新失败')
  } finally {
    graphToggleLoading.value = false
  }
}

async function removeDoc(docId) {
  try {
    await kbStore.removeDocument(kbId, docId)
    await loadDocs()
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

async function selectDoc(doc) {
  selectedDocId.value = doc.id
  chunksLoading.value = true
  try {
    docChunks.value = await listDocumentChunks(kbId, doc.id)
  } finally {
    chunksLoading.value = false
  }
}

async function openDocChunks(doc) {
  activeTab.value = 'chunks'
  await selectDoc(doc)
}

function openChunkDetail(chunk) {
  selectedChunk.value = chunk
  chunkModalOpen.value = true
}

const groupedQueryResults = computed(() => {
  const map = new Map()
  queryResults.value.forEach((r, i) => {
    const filename = r.metadata?.source_path?.split('/').pop() || r.metadata?.source_path || '未知来源'
    if (!map.has(filename)) map.set(filename, [])
    map.get(filename).push({ ...r, _rank: i + 1 })
  })
  return Array.from(map.entries()).map(([filename, chunks]) => ({ filename, chunks }))
})

function toggleGroup(filename) {
  if (collapsedGroups.has(filename)) collapsedGroups.delete(filename)
  else collapsedGroups.add(filename)
}

function openResultDetail(r) {
  selectedResult.value = r
  resultDetailOpen.value = true
}

async function runQuery() {
  if (!queryText.value.trim()) return
  queryLoading.value = true
  queryDone.value = false
  queryResults.value = []
  try {
    const res = await testQuery(kbId, queryText.value, queryTopK.value, queryMode.value)
    queryResults.value = res.results || []
    queryDone.value = true
  } catch (e) {
    message.error(e.message || '检索失败')
  } finally {
    queryLoading.value = false
  }
}

// ── Eval logic ──

function syncRagas() {
  localRagasGen.value  = settingsStore.ragasGeneratorModel
  localRagasEval.value = settingsStore.ragasEvaluatorModel
}

async function saveRagasSettings() {
  ragasSaving.value = true
  try {
    await settingsStore.saveRagasSettings({
      generatorModel: localRagasGen.value,
      evaluatorModel: localRagasEval.value,
    })
    message.success('评估配置已保存')
  } catch (e) {
    message.error('保存失败：' + (e.message || ''))
  } finally {
    ragasSaving.value = false
  }
}

async function loadEvalData() {
  await Promise.all([loadDatasets(), loadRuns()])
}

async function loadDatasets() {
  datasetsLoading.value = true
  try { datasets.value = await listDatasets(kbId) }
  finally { datasetsLoading.value = false }
}

async function loadRuns() {
  runsLoading.value = true
  try { evalRuns.value = await listEvalRuns(kbId) }
  finally { runsLoading.value = false }
}

function startEvalPoll() {
  evalPollTimer = setInterval(() => {
    if (evalRuns.value.some(r => r.status === 'running')) loadRuns()
  }, 3000)
}

function stopEvalPoll() {
  if (evalPollTimer) { clearInterval(evalPollTimer); evalPollTimer = null }
}

async function handleDatasetUpload(file) {
  const name = file.name.replace(/\.jsonl$/, '')
  try {
    await uploadDataset(kbId, name, file)
    await loadDatasets()
    message.success('数据集上传成功')
  } catch (e) {
    message.error(e.message || '上传失败')
  }
  return false
}

async function submitGenerateDataset() {
  generatingDataset.value = true
  try {
    await generateDataset(kbId, {
      name: generateForm.value.name || undefined,
      n_questions: generateForm.value.n_questions,
      simple_ratio: generateForm.value.simple_ratio,
      multi_context_ratio: generateForm.value.multi_context_ratio,
      reasoning_ratio: Math.max(0, 1 - generateForm.value.simple_ratio - generateForm.value.multi_context_ratio),
    })
    generateModalOpen.value = false
    message.success('数据集生成已启动，请稍后刷新')
    setTimeout(() => loadDatasets(), 3000)
  } catch (e) {
    message.error(e.message || '生成失败')
  } finally {
    generatingDataset.value = false
  }
}

async function removeDataset(id) {
  try {
    await deleteDataset(id)
    datasets.value = datasets.value.filter(d => d.id !== id)
    if (selectedDataset.value?.id === id) selectedDataset.value = null
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

async function startRun(ds) {
  startingRun.value = ds.id + ':quick'
  try {
    const run = await createEvalRun(kbId, { dataset_id: ds.id, name: `${ds.name} - Quick`, top_k: 5 })
    evalRuns.value.unshift(run)
    if (!evalPollTimer) startEvalPoll()
    message.success('Quick 评估已启动')
  } catch (e) {
    message.error(e.message || '启动失败')
  } finally {
    startingRun.value = null
  }
}

async function startRagasRun(ds) {
  startingRun.value = ds.id + ':ragas'
  try {
    const run = await createRagasRun(kbId, { dataset_id: ds.id, name: `${ds.name} - RAGAS`, top_k: 5 })
    evalRuns.value.unshift(run)
    if (!evalPollTimer) startEvalPoll()
    message.success('RAGAS 评估已启动')
  } catch (e) {
    message.error(e.message || '启动失败')
  } finally {
    startingRun.value = null
  }
}

async function startAgentRun(ds) {
  startingRun.value = ds.id + ':agent'
  try {
    const run = await createAgentRun(kbId, { dataset_id: ds.id, name: `${ds.name} - Agent`, top_k: 5 })
    evalRuns.value.unshift(run)
    if (!evalPollTimer) startEvalPoll()
    message.success('Agent 评估已启动')
  } catch (e) {
    message.error(e.message || '启动失败')
  } finally {
    startingRun.value = null
  }
}

async function toggleRun(run) {
  if (expandedRun.value === run.id) { expandedRun.value = null; return }
  expandedRun.value = run.id
  expandedRunType.value = run.eval_type || 'quick'
  itemsLoading.value = true
  try {
    const detail = await getEvalRun(kbId, run.id)
    runItems.value = detail.items || []
  } finally {
    itemsLoading.value = false
  }
}

async function removeRun(run) {
  try {
    await deleteEvalRun(kbId, run.id)
    evalRuns.value = evalRuns.value.filter(r => r.id !== run.id)
    if (expandedRun.value === run.id) expandedRun.value = null
    message.success('已删除')
  } catch (e) {
    message.error(e.message || '删除失败')
  }
}

function runProgress(run) {
  if (!run.total_items) return 0
  return Math.round((run.completed_items / run.total_items) * 100)
}

function runStatusColor(s) {
  return { pending: 'default', running: 'blue', completed: 'green', failed: 'red' }[s] || 'default'
}

function runStatusLabel(s) {
  return { pending: '待运行', running: '运行中', completed: '完成', failed: '失败' }[s] || s
}

function metricColor(v) {
  if (v === undefined) return ''
  if (v >= 0.7) return 'metric-high'
  if (v >= 0.4) return 'metric-mid'
  return 'metric-low'
}

function fmtDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}

// ── Shared helpers ──

function contentTypeColor(type) {
  return { code: 'orange', table: 'green', list: 'cyan', text: 'default' }[type] || 'default'
}

const STRUCTURED_META_KEYS = ['title', 'section_path', 'section_level', 'content_type', 'chunk_strategy_used', 'page_num', 'tags', 'summary', 'refined_by', 'enriched_by', 'prev_chunk_id', 'next_chunk_id']

function structuredMetaFields(meta) {
  const result = {}
  for (const key of STRUCTURED_META_KEYS) {
    if (meta[key] != null) {
      const val = Array.isArray(meta[key]) ? meta[key].join(', ') : String(meta[key])
      if (val) result[key] = val
    }
  }
  return result
}

function renderMarkdown(text) {
  return marked.parse(text || '')
}

function rawMetaRemainder(meta) {
  const knownKeys = new Set([...STRUCTURED_META_KEYS, 'source_path', 'source_ref', 'chunk_index', 'doc_type'])
  const remainder = {}
  for (const [k, v] of Object.entries(meta)) {
    if (!knownKeys.has(k)) remainder[k] = v
  }
  return remainder
}

function statusColor(s) {
  return { uploaded: 'default', parsing: 'orange', indexing: 'blue', indexed: 'green', error: 'red' }[s] || 'default'
}

function statusLabel(s) {
  return { uploaded: '待处理', parsing: '解析中', indexing: '索引中', indexed: '已索引', error: '错误' }[s] || s
}

function isIndexing(s) { return s === 'parsing' || s === 'indexing' }

function formatSize(bytes) {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<style scoped>
.kb-detail-page { padding: 32px; }
.page-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.header-left { display: flex; align-items: center; }
.header-left h2 { font-size: 22px; font-weight: 700; margin: 0; }
.header-right { display: flex; align-items: center; gap: 12px; }
.stat-chip { font-size: 13px; color: var(--nr-ink-2); display: flex; align-items: center; gap: 4px; }

.tab-toolbar { margin-bottom: 16px; }

/* Chunk browser */
.chunk-browser { display: flex; gap: 0; border: 1px solid var(--nr-border); border-radius: 8px; overflow: hidden; height: 560px; }

.chunk-doc-list {
  width: 220px;
  border-right: 1px solid var(--nr-border);
  overflow-y: auto;
  background: var(--nr-rail);
}
.chunk-doc-item {
  display: flex; align-items: center; gap: 8px; padding: 10px 14px;
  cursor: pointer; font-size: 13px; transition: background 0.15s;
  border-bottom: 1px solid var(--nr-border);
}
.chunk-doc-item:hover { background: var(--nr-clay-tint); }
.chunk-doc-item.active { background: var(--nr-clay-soft); font-weight: 600; color: var(--nr-clay); }
.chunk-doc-name { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

.chunk-list-panel { flex: 1; overflow-y: auto; padding: 12px; }
.chunk-placeholder { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--nr-ink-3); font-size: 14px; }

.chunk-items { display: flex; flex-direction: column; gap: 8px; }
.chunk-item {
  border: 1px solid var(--nr-border); border-radius: 6px; padding: 10px 12px;
  cursor: pointer; transition: all 0.15s;
}
.chunk-item:hover { border-color: var(--nr-clay-soft); background: var(--nr-clay-tint); }
.chunk-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.chunk-index { font-size: 11px; font-weight: 700; color: var(--nr-clay); background: var(--nr-clay-soft); border-radius: 4px; padding: 1px 6px; }
.chunk-tokens { font-size: 11px; color: var(--nr-ink-3); }
.chunk-section-path { font-size: 11px; color: var(--nr-ink-2); margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chunk-preview { font-size: 12px; color: var(--nr-ink-2); line-height: 1.5; }

/* Query */
.query-panel { max-width: 900px; }
.query-input-row { display: flex; gap: 8px; margin-bottom: 20px; }
.query-results { display: flex; flex-direction: column; gap: 8px; }

.result-file-group { border: 1px solid var(--nr-border); border-radius: 8px; overflow: hidden; }
.result-file-header {
  display: flex; align-items: center; gap: 8px; padding: 10px 14px;
  background: var(--nr-rail); cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--nr-border);
}
.result-file-header:hover { background: var(--nr-clay-tint); }
.result-file-name { flex: 1; font-size: 13px; font-weight: 600; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.group-toggle-icon { font-size: 12px; color: var(--nr-ink-3); transition: transform 0.2s; }
.group-toggle-icon.expanded { transform: rotate(90deg); }
.result-file-chunks { display: flex; flex-direction: column; gap: 0; }
.query-result-item { padding: 12px 14px; border-bottom: 1px solid var(--nr-rail); cursor: pointer; transition: background 0.15s; }
.query-result-item:hover { background: var(--nr-clay-tint); }
.query-result-item:last-child { border-bottom: none; }
.result-header { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.result-rank { font-size: 12px; font-weight: 700; color: var(--nr-clay); }
.result-text { font-size: 13px; color: var(--nr-ink-2); line-height: 1.6; }

/* Eval tab */
.eval-tab { max-width: 1100px; }
.eval-layout { display: grid; grid-template-columns: 300px 1fr; gap: 24px; align-items: start; }

.panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.panel-header h3 { font-size: 15px; font-weight: 700; margin: 0; }

.datasets-panel { background: #fff; border: 1px solid var(--nr-border); border-radius: 10px; padding: 16px; }
.dataset-list { display: flex; flex-direction: column; gap: 8px; }
.dataset-item {
  border: 1px solid var(--nr-border); border-radius: 8px; padding: 10px 12px;
  cursor: pointer; transition: all 0.15s;
}
.dataset-item:hover { border-color: var(--nr-clay-soft); background: var(--nr-clay-tint); }
.dataset-item.active { border-color: var(--nr-clay); background: var(--nr-clay-soft); }
.ds-name { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.ds-meta { font-size: 11px; color: var(--nr-ink-3); margin-bottom: 8px; }
.ds-actions { display: flex; align-items: center; gap: 6px; }
.jsonl-hint { font-size: 11px; color: var(--nr-ink-3); margin-top: 12px; line-height: 1.6; }
.jsonl-hint code { background: var(--nr-rail); padding: 1px 4px; border-radius: 3px; font-size: 10px; }

.runs-panel { background: #fff; border: 1px solid var(--nr-border); border-radius: 10px; padding: 16px; }
.runs-list { display: flex; flex-direction: column; gap: 10px; }
.run-card { border: 1px solid var(--nr-border); border-radius: 8px; overflow: hidden; }
.run-card.expanded { border-color: var(--nr-clay-soft); }
.run-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 14px; cursor: pointer; transition: background 0.15s;
}
.run-header:hover { background: var(--nr-rail); }
.run-left { display: flex; align-items: center; gap: 8px; }
.run-name { font-size: 13px; font-weight: 600; }
.run-right { display: flex; align-items: center; gap: 10px; }
.run-score { font-size: 13px; font-weight: 700; color: var(--nr-clay); }
.run-metrics { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 10px; }
.metric-chip { font-size: 11px; background: var(--nr-clay-tint); border: 1px solid var(--nr-clay-soft); border-radius: 4px; padding: 2px 8px; color: var(--nr-clay); }
.run-items { padding: 0 14px 12px; border-top: 1px solid var(--nr-border); }
.items-table { font-size: 12px; }
.items-header { display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--nr-border); font-weight: 600; color: var(--nr-ink-2); }
.items-row { display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--nr-rail); align-items: flex-start; }
.item-text { overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.metric-high { color: var(--nr-sage); font-weight: 600; }
.metric-mid  { color: var(--nr-gold); }
.metric-low  { color: var(--nr-danger); }

/* Chunk / result modal */
.chunk-meta-pills { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.meta-pill { font-size: 12px; color: var(--nr-ink-2); background: var(--nr-rail); border-radius: 4px; padding: 2px 8px; }
.chunk-content { background: var(--nr-rail); border-radius: 6px; padding: 12px 16px; font-size: 13px; line-height: 1.7; max-height: 400px; overflow-y: auto; word-break: break-word; }
.chunk-content :deep(h1),.chunk-content :deep(h2),.chunk-content :deep(h3) { font-weight: 600; margin: 8px 0 4px; }
.chunk-content :deep(h1) { font-size: 16px; }
.chunk-content :deep(h2) { font-size: 14px; }
.chunk-content :deep(h3) { font-size: 13px; }
.chunk-content :deep(p) { margin: 4px 0; }
.chunk-content :deep(pre) { background: var(--nr-border); border-radius: 4px; padding: 8px; overflow-x: auto; font-size: 12px; }
.chunk-content :deep(code) { background: var(--nr-border); border-radius: 3px; padding: 1px 4px; font-size: 12px; }
.chunk-content :deep(pre code) { background: none; padding: 0; }
.chunk-content :deep(table) { border-collapse: collapse; width: 100%; font-size: 12px; margin: 6px 0; }
.chunk-content :deep(th),.chunk-content :deep(td) { border: 1px solid var(--nr-border-strong); padding: 4px 8px; }
.chunk-content :deep(th) { background: var(--nr-border); }
.chunk-content :deep(ul),.chunk-content :deep(ol) { padding-left: 20px; margin: 4px 0; }
.chunk-content :deep(blockquote) { border-left: 3px solid var(--nr-ink-3); margin: 4px 0; padding-left: 10px; color: var(--nr-ink-2); }
.chunk-meta-title { font-size: 13px; font-weight: 600; margin: 12px 0 6px; }
.chunk-meta-structured { border: 1px solid var(--nr-border); border-radius: 6px; overflow: hidden; }
.meta-row { display: flex; gap: 0; border-bottom: 1px solid var(--nr-rail); font-size: 12px; }
.meta-row:last-child { border-bottom: none; }
.meta-key { width: 160px; flex-shrink: 0; padding: 6px 10px; background: var(--nr-rail); color: var(--nr-ink-2); font-family: monospace; }
.meta-val { flex: 1; padding: 6px 10px; color: var(--nr-ink); word-break: break-all; }
.meta-row-json { align-items: flex-start; }
.meta-val-json { flex: 1; padding: 6px 10px; font-size: 11px; margin: 0; background: none; white-space: pre-wrap; word-break: break-word; max-height: 160px; overflow-y: auto; }

/* PDF preview modal */
.pdf-modal-title { display: flex; align-items: center; gap: 10px; width: 100%; }
.pdf-modal-filename { font-size: 14px; font-weight: 600; color: var(--nr-ink); overflow: hidden; white-space: nowrap; text-overflow: ellipsis; max-width: 500px; }
.pdf-modal-count { font-size: 12px; color: var(--nr-ink-3); flex-shrink: 0; }

.pdf-viewer-body { display: flex; flex: 1; overflow: hidden; height: 100%; }

.pdf-left {
  flex: 1;
  border-right: 1px solid var(--nr-border);
  display: flex;
  flex-direction: column;
  background: var(--nr-border);
  overflow: hidden;
}
.pdf-embed { flex: 1; overflow-y: auto; }
.pdf-embed :deep(.vue-pdf-embed__page) { margin: 0 auto; display: block; }
.pdf-embed :deep(.textLayer) { position: absolute; }
.pdf-nav {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  padding: 6px 0; background: #fff; border-top: 1px solid var(--nr-border); flex-shrink: 0;
}
.pdf-nav-label { font-size: 13px; color: var(--nr-ink-2); min-width: 60px; text-align: center; }
.pdf-loading { display: flex; align-items: center; justify-content: center; height: 100%; }
.pdf-error { display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--nr-ink-3); text-align: center; padding: 32px; }

.pdf-right {
  width: 340px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--nr-rail);
  overflow: hidden;
}
.pdf-chunk-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}
.pdf-chunk-card {
  border: 1px solid var(--nr-border);
  border-radius: 6px;
  padding: 8px 10px;
  background: #fff;
  cursor: pointer;
  transition: all 0.15s;
  flex-shrink: 0;
}
.pdf-chunk-card:hover { border-color: var(--nr-clay-soft); background: var(--nr-clay-tint); }
.pdf-chunk-card.active { border-color: var(--nr-clay); background: var(--nr-clay-soft); box-shadow: 0 0 0 2px rgba(22,119,255,0.15); }
.pdf-chunk-header { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
.pdf-chunk-index { font-size: 11px; font-weight: 700; color: var(--nr-clay); background: var(--nr-clay-soft); border-radius: 4px; padding: 1px 6px; }
.pdf-chunk-tokens { font-size: 11px; color: var(--nr-ink-3); margin-left: auto; }
.pdf-chunk-preview {
  font-size: 11px; color: var(--nr-ink-2); line-height: 1.4;
  overflow: hidden; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}

/* Bottom: markdown preview of selected chunk */
.pdf-chunk-md {
  flex-shrink: 0;
  height: 40%;
  min-height: 120px;
  border-top: 1px solid var(--nr-border);
  display: flex;
  flex-direction: column;
  background: #fff;
  overflow: hidden;
}
.pdf-chunk-md-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; font-size: 12px; font-weight: 600; color: var(--nr-ink-2);
  background: var(--nr-rail); border-bottom: 1px solid var(--nr-border); flex-shrink: 0;
}
.pdf-chunk-md-tokens { font-size: 11px; color: var(--nr-ink-3); font-weight: 400; }
.pdf-chunk-md-body {
  flex: 1; overflow-y: auto; padding: 10px 14px;
  font-size: 12px; line-height: 1.6;
}

/* Chunks grid view */
.chunks-grid-panel {
  flex: 1; overflow-y: auto; padding: 16px;
}
.chunks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.chunks-grid-card {
  border: 1px solid var(--nr-border); border-radius: 8px; padding: 12px;
  cursor: pointer; transition: all 0.15s; background: #fff;
}
.chunks-grid-card:hover { border-color: var(--nr-clay-soft); box-shadow: 0 2px 8px rgba(22,119,255,0.1); }
.chunks-grid-header { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
.chunks-grid-content {
  font-size: 12px; color: var(--nr-ink-2); line-height: 1.5;
  overflow: hidden; display: -webkit-box;
  -webkit-line-clamp: 4; -webkit-box-orient: vertical;
}

/* Full text preview modal */
.fulltext-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
  font-size: 14px;
  line-height: 1.8;
}
.fulltext-body :deep(h1) { font-size: 20px; font-weight: 700; margin: 16px 0 8px; }
.fulltext-body :deep(h2) { font-size: 17px; font-weight: 600; margin: 14px 0 6px; }
.fulltext-body :deep(h3) { font-size: 15px; font-weight: 600; margin: 12px 0 4px; }
.fulltext-body :deep(p) { margin: 6px 0; }
.fulltext-body :deep(hr) { border: none; border-top: 1px dashed var(--nr-border); margin: 16px 0; }
.fulltext-body :deep(pre) { background: var(--nr-rail); border-radius: 6px; padding: 12px; overflow-x: auto; font-size: 12px; }
.fulltext-body :deep(code) { background: var(--nr-border); border-radius: 3px; padding: 1px 5px; font-size: 12px; }
.fulltext-body :deep(pre code) { background: none; padding: 0; }
.fulltext-body :deep(table) { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
.fulltext-body :deep(th), .fulltext-body :deep(td) { border: 1px solid var(--nr-border-strong); padding: 6px 10px; }
.fulltext-body :deep(th) { background: var(--nr-rail); font-weight: 600; }
.fulltext-body :deep(ul), .fulltext-body :deep(ol) { padding-left: 24px; margin: 4px 0; }
.fulltext-body :deep(blockquote) { border-left: 3px solid var(--nr-border-strong); margin: 6px 0; padding: 4px 12px; color: var(--nr-ink-2); }
.fulltext-body :deep(img) { max-width: 100%; border-radius: 4px; margin: 8px 0; }

/* Wiki entity browser */
.wiki-wrap { display: flex; gap: 16px; }
.wiki-list { width: 260px; flex-shrink: 0; max-height: 70vh; overflow-y: auto; border-right: 1px solid #eee; padding-right: 8px; }
.wiki-ent { display: flex; justify-content: space-between; padding: 6px 8px; border-radius: 6px; cursor: pointer; }
.wiki-ent:hover { background: #f5f5f5; }
.wiki-ent.active { background: #e6f0ff; }
.wiki-ent-count { color: #999; font-size: 12px; }
.wiki-detail { flex: 1; min-width: 0; }
.wiki-title { display: flex; align-items: center; gap: 8px; }
.wiki-sub { color: #999; font-size: 13px; font-weight: normal; }
.wiki-fact { border: 1px solid #eee; border-radius: 6px; margin-bottom: 6px; }
.wiki-fact-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; cursor: pointer; }
.wiki-fact-row em { color: #C15F3C; font-style: normal; }
.wiki-evidence { border-top: 1px dashed #eee; padding: 8px 10px; background: #fafafa; }
.wiki-chunk { margin-bottom: 8px; }
.wiki-chunk-src { font-size: 12px; color: #888; }
.wiki-chunk-text { font-size: 13px; white-space: pre-wrap; }
.wiki-neighbor { cursor: pointer; margin-bottom: 6px; }
.wiki-empty { padding: 40px 0; }
.wiki-article { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #f0f0f0; }
.wiki-article-meta { margin-top: 8px; display: flex; align-items: center; gap: 8px; }
</style>
