import { useUserStore } from '@/stores/user'

// Long-lived SSE subscription to a conversation's live stream
// (`/api/conversations/{id}/live`). Carries server-pushed messages that this tab did NOT
// initiate — currently cron results delivered into the conversation. Mirrors useRunStream's
// fetch-stream parsing; heartbeat lines (": ...") are naturally ignored (not "data:").
export function useConversationLive() {
  let controller = null

  async function start(convId, { onMessage } = {}) {
    stop()
    controller = new AbortController()
    const userStore = useUserStore()

    try {
      const response = await fetch(`/api/conversations/${convId}/live`, {
        headers: userStore.getAuthHeaders(),
        signal: controller.signal,
      })
      if (!response.ok || !response.body) return

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let dataLines = []

      const dispatch = () => {
        if (!dataLines.length) return
        const text = dataLines.join('\n')
        dataLines = []
        try {
          const event = JSON.parse(text)
          if (event.type === 'cron_message') onMessage?.(event)
        } catch (e) {
          console.warn('conv-live parse error:', e, text)
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const raw of lines) {
          const line = raw.replace(/\r$/, '')
          if (!line) { dispatch(); continue }
          if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') console.error('conv-live stream error:', err)
    }
  }

  function stop() {
    controller?.abort()
    controller = null
  }

  return { start, stop }
}
