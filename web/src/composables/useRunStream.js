import { useUserStore } from '@/stores/user'

export function useRunStream() {
  let controller = null

  async function start(runId, { onDelta, onToolHint, onEnd, onToolCall, onMessageComplete, onSubagentResult, onCitations } = {}) {
    console.log('[SSE] start v3', runId)
    stop()
    controller = new AbortController()
    const userStore = useUserStore()

    try {
      const response = await fetch(`/api/runs/${runId}/events`, {
        headers: userStore.getAuthHeaders(),
        signal: controller.signal
      })

      if (!response.ok || !response.body) {
        onEnd?.('failed')
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let dataLines = []

      const dispatch = () => {
        if (!dataLines.length) return
        const text = dataLines.join('\n')
        try {
          const event = JSON.parse(text)
          console.log('[SSE] event:', event.type)
          if (event.type === 'message_delta') onDelta?.(event.chunk)
          else if (event.type === 'tool_hint') onToolHint?.(event.content)
          else if (event.type === 'tool_call') onToolCall?.(event)
          else if (event.type === 'message_complete') { onMessageComplete?.() }
          else if (event.type === 'subagent_result') onSubagentResult?.(event)
          else if (event.type === 'citations') onCitations?.(event)
          else if (event.type === 'run_end') onEnd?.(event.status)
        } catch (e) {
          console.warn('SSE parse error:', e, text)
        }
        dataLines = []
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        if (chunk.includes('message_complete') || chunk.includes('heartbeat')) {
          console.log('[SSE] raw chunk:', JSON.stringify(chunk))
        }
        buffer += chunk
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const raw of lines) {
          const line = raw.replace(/\r$/, '')
          if (!line) { dispatch(); continue }
          if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
        }
      }
      dispatch()
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('SSE stream error:', err)
        onEnd?.('failed')
      }
    }
  }

  function stop() {
    controller?.abort()
    controller = null
  }

  return { start, stop }
}
